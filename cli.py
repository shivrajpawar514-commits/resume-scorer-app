"""
CLI Resume Scorer & Batch Evaluator
Run automated resume evaluations and batch audits directly from the command line.
"""

import os
import sys
import csv
import json
import argparse
from typing import Optional, List, Dict, Any

from analyzer.skills_taxonomy import ROLE_BENCHMARKS
from analyzer.extractor import extract_text_from_file
from analyzer.scoring_engine import score_resume
from analyzer.report_builder import generate_pdf_report, generate_markdown_report
from analyzer.sample_resumes import SAMPLE_RESUMES

def print_banner():
    print("=" * 75)
    print(" ⚡ AI/ML Resume Scorer & ATS Analytics Engine (CLI Mode)")
    print("=" * 75)

def evaluate_single_file(file_path: str, role_name: Optional[str] = None, jd_path: Optional[str] = None) -> dict:
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "rb") as f:
        text = extract_text_from_file(f)

    if not text.strip():
        print(f"❌ Error: Could not extract readable text from {file_path}")
        sys.exit(1)

    jd_text = None
    if jd_path and os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()

    return score_resume(resume_text=text, target_role_name=role_name, custom_jd_text=jd_text)

def export_batch_to_csv(batch_results: List[Dict[str, Any]], target_file: str):
    """Write batch scoring results and 5D breakdowns into CSV format."""
    fieldnames = [
        "Rank", "Filename", "Composite Score (%)", "Verdict",
        "Hard Skills Score", "Semantic Fit Score", "Quant Impact Score",
        "MLOps Readiness Score", "ATS Health Score",
        "Matched Core Skills", "Missing Core Skills", "Top Strong Verbs"
    ]
    with open(target_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(batch_results, 1):
            res = item.get("results", {})
            b = res.get("scores_breakdown", {})
            sk = res.get("skills_summary", {})
            imp = res.get("impact_analysis", {})
            writer.writerow({
                "Rank": i,
                "Filename": item.get("filename", "Unknown"),
                "Composite Score (%)": res.get("composite_score", 0),
                "Verdict": res.get("verdict", "N/A"),
                "Hard Skills Score": b.get("hard_skill_score", 0),
                "Semantic Fit Score": b.get("semantic_score", 0),
                "Quant Impact Score": b.get("impact_score", 0),
                "MLOps Readiness Score": b.get("mlops_score", 0),
                "ATS Health Score": b.get("ats_health_score", 0),
                "Matched Core Skills": "; ".join(sk.get("matched_core", [])),
                "Missing Core Skills": "; ".join(sk.get("missing_core", [])),
                "Top Strong Verbs": "; ".join(imp.get("strong_verbs_found", [])[:5]),
            })

def main():
    parser = argparse.ArgumentParser(description="AI/ML Resume Scorer & ATS Intelligence CLI")
    parser.add_argument("--resume", "-r", type=str, help="Path to resume file (PDF, DOCX, TXT)")
    parser.add_argument("--role", type=str, default=list(ROLE_BENCHMARKS.keys())[0], help="Target AI/ML Role Benchmark")
    parser.add_argument("--jd", type=str, help="Path to custom Job Description text file")
    parser.add_argument("--sample", type=str, help="Name of sample resume to test (or 'list' to see available samples)")
    parser.add_argument("--json-out", type=str, help="Export results to JSON file")
    parser.add_argument("--csv-out", type=str, help="Export batch or single results to CSV leaderboard")
    parser.add_argument("--pdf-out", type=str, help="Export results to PDF audit report")
    parser.add_argument("--md-out", type=str, help="Export results to Markdown audit report")
    parser.add_argument("--batch-dir", type=str, help="Path to directory containing multiple resumes for batch scoring")

    args = parser.parse_args()
    print_banner()

    if args.sample:
        if args.sample == "list":
            print("\nAvailable Preloaded Sample Resumes:")
            for s in SAMPLE_RESUMES.keys():
                print(f"  - {s}")
            return
        
        # Match sample
        matched_sample = None
        for k in SAMPLE_RESUMES.keys():
            if args.sample.lower() in k.lower():
                matched_sample = k
                break
        if not matched_sample:
            print(f"❌ Sample not found: {args.sample}. Use --sample list to view options.")
            return

        print(f"📄 Testing Preloaded Sample: {matched_sample}")
        results = score_resume(SAMPLE_RESUMES[matched_sample], target_role_name=args.role)
        display_results(results)
        handle_exports(results, args, filename=matched_sample)
        return

    if args.batch_dir:
        if not os.path.exists(args.batch_dir):
            print(f"❌ Error: Batch directory not found: {args.batch_dir}")
            return

        files = [os.path.join(args.batch_dir, f) for f in os.listdir(args.batch_dir) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
        if not files:
            print(f"No resume files found in {args.batch_dir}")
            return

        print(f"\n🚀 Running batch evaluation on {len(files)} resumes against '{args.role}'...\n")
        print(f"{'Rank':<5} | {'Filename':<32} | {'Score':<8} | {'Verdict'}")
        print("-" * 75)
        
        batch_results = []
        for file_path in files:
            fname = os.path.basename(file_path)
            try:
                res = evaluate_single_file(file_path, role_name=args.role, jd_path=args.jd)
                batch_results.append({"filename": fname, "results": res})
            except Exception as e:
                print(f"{'-':<5} | {fname[:30]:<32} | ERROR    | {str(e)[:25]}")

        # Sort batch by score descending
        batch_results.sort(key=lambda x: x["results"].get("composite_score", 0), reverse=True)

        for rank, item in enumerate(batch_results, 1):
            fname = item["filename"]
            res = item["results"]
            print(f"#{rank:<4} | {fname[:30]:<32} | {res['composite_score']:<7}% | {res['verdict']}")

        if batch_results:
            scores = [x["results"]["composite_score"] for x in batch_results]
            avg_score = round(sum(scores) / len(scores), 1)
            print("-" * 75)
            print(f"📈 Batch Summary: {len(batch_results)} Evaluated | Average Score: {avg_score}% | Top Score: {max(scores)}%")

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(batch_results, f, indent=2)
            print(f"\n✓ Saved batch results to JSON: {args.json_out}")

        if args.csv_out:
            export_batch_to_csv(batch_results, args.csv_out)
            print(f"✓ Saved batch leaderboard to CSV: {args.csv_out}")
        return

    if not args.resume:
        print("ℹ️ No resume specified. Running evaluation on default Senior GenAI sample...")
        default_sample = list(SAMPLE_RESUMES.keys())[0]
        results = score_resume(SAMPLE_RESUMES[default_sample], target_role_name=args.role)
        source_name = "default_sample"
    else:
        print(f"📄 Analyzing: {args.resume}")
        results = evaluate_single_file(args.resume, role_name=args.role, jd_path=args.jd)
        source_name = os.path.basename(args.resume)

    display_results(results)
    handle_exports(results, args, filename=source_name)

def display_results(results: dict):
    b = results["scores_breakdown"]
    sk = results["skills_summary"]
    imp = results["impact_analysis"]

    print("\n" + "=" * 75)
    print(f" 🎯 TARGET BENCHMARK: {results['target_role']}")
    print(f" 🏆 OVERALL SCORE:   {results['composite_score']}%  ({results['verdict']})")
    print("=" * 75)
    print("\n📊 5D Score Breakdown:")
    print(f"  • Hard Skills Match:       {b['hard_skill_score']}%")
    print(f"  • TF-IDF Semantic Fit:     {b['semantic_score']}%")
    print(f"  • Quantifiable Impact:     {b['impact_score']}%")
    print(f"  • MLOps & Production:      {b['mlops_score']}%")
    print(f"  • ATS Health & Structure:  {b['ats_health_score']}%")

    print("\n🧠 Skill Alignment:")
    print(f"  • Matched Core ({len(sk['matched_core'])}): {', '.join(sk['matched_core'][:10])}")
    print(f"  • Missing Core ({len(sk['missing_core'])}): {', '.join(sk['missing_core'][:8]) or 'None'}")
    print(f"  • Total Extracted Skills: {sk['total_candidate_skills']}")

    print("\n💥 Impact Analytics:")
    print(f"  • Action Verbs Found:     {', '.join(imp['strong_verbs_found'][:8]) or 'None'}")
    print(f"  • Quant Metrics Found:    {', '.join(imp['metrics_found'][:6]) or 'None'}")
    print(f"  • Strong X-Y-Z Bullets:   {imp['strong_bullets_count']} / {imp['total_bullets_analyzed']}")
    print("=" * 75 + "\n")

def handle_exports(results: dict, args, filename: str = "resume"):
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Exported JSON report: {args.json_out}")

    if args.csv_out:
        export_batch_to_csv([{"filename": filename, "results": results}], args.csv_out)
        print(f"✓ Exported CSV report: {args.csv_out}")

    if args.pdf_out:
        pdf_bytes = generate_pdf_report(results)
        with open(args.pdf_out, "wb") as f:
            f.write(pdf_bytes)
        print(f"✓ Exported PDF report: {args.pdf_out}")

    if args.md_out:
        md_text = generate_markdown_report(results)
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"✓ Exported Markdown report: {args.md_out}")

if __name__ == "__main__":
    main()
