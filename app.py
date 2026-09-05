"""
AI/ML Resume Scorer & Advanced Analytics Engine
Streamlit Application with Modern Glassmorphism UI, Multi-Tier NLP, ATS Scoring, and Interactive Visualizations.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from analyzer.skills_taxonomy import ROLE_BENCHMARKS, CATEGORIZED_SKILLS, SKILL_LEARNING_RESOURCES
from analyzer.extractor import extract_text_from_file, extract_contact_info
from analyzer.section_parser import parse_resume_sections
from analyzer.matcher import extract_skills_from_text, extract_skills_from_job_description
from analyzer.impact_analyzer import analyze_impact_and_verbs, STRONG_ACTION_VERBS, WEAK_PASSIVE_PHRASES
from analyzer.scoring_engine import score_resume
from analyzer.report_builder import generate_pdf_report, generate_markdown_report
from analyzer.sample_resumes import SAMPLE_RESUMES

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI/ML Resume Intelligence & ATS Scorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Elegant Custom Design System & CSS
# ---------------------------------------------------------
def inject_custom_css(dark_mode: bool = True):
    if dark_mode:
        bg_canvas = "#090D16"
        bg_surface = "rgba(17, 24, 39, 0.85)"
        bg_surface_subtle = "rgba(31, 41, 55, 0.65)"
        border_subtle = "rgba(255, 255, 255, 0.08)"
        border_accent = "rgba(99, 102, 241, 0.35)"
        text_main = "#F9FAFB"
        text_secondary = "#9CA3AF"
        text_accent = "#818CF8"
        shadow_card = "0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 1px 1px rgba(255, 255, 255, 0.05)"
        gradient_brand = "linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%)"
        gradient_pill = "linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)"
    else:
        bg_canvas = "#F8FAFC"
        bg_surface = "rgba(255, 255, 255, 0.95)"
        bg_surface_subtle = "rgba(241, 245, 249, 0.85)"
        border_subtle = "rgba(226, 232, 240, 0.8)"
        border_accent = "rgba(99, 102, 241, 0.3)"
        text_main = "#0F172A"
        text_secondary = "#64748B"
        text_accent = "#4F46E5"
        shadow_card = "0 10px 25px -5px rgba(15, 23, 42, 0.06), 0 0 1px 1px rgba(15, 23, 42, 0.04)"
        gradient_brand = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #DB2777 100%)"
        gradient_pill = "linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background-color: {bg_canvas};
        color: {text_main};
    }}

    /* Hero Typography */
    .hero-container {{
        padding: 0.5rem 0 1.5rem 0;
        margin-bottom: 0.5rem;
    }}

    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {gradient_pill};
        border: 1px solid {border_accent};
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: {text_accent};
        margin-bottom: 0.8rem;
    }}

    .hero-title {{
        background: {gradient_brand};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
    }}

    .hero-desc {{
        color: {text_secondary};
        font-size: 1.05rem;
        line-height: 1.5;
        max-width: 800px;
    }}

    /* Card Containers */
    .card-clean {{
        background: {bg_surface};
        border: 1px solid {border_subtle};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: {shadow_card};
        margin-bottom: 1.2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease, border-color 0.2s ease;
    }}

    .card-clean:hover {{
        border-color: {border_accent};
        transform: translateY(-2px);
    }}

    /* Metric Banner Boxes */
    .metric-subcard {{
        background: {bg_surface_subtle};
        border: 1px solid {border_subtle};
        border-radius: 12px;
        padding: 1rem 1.1rem;
        text-align: left;
    }}

    .metric-subcard-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 0.3rem;
    }}

    .metric-subcard-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: {text_main};
        line-height: 1.1;
    }}

    /* Custom Skill Pill Badges */
    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 12px;
        margin: 3px 4px 3px 0;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
    }}

    .pill-matched {{
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }}

    .pill-missing {{
        background: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.28);
    }}

    .pill-secondary {{
        background: rgba(59, 130, 246, 0.12);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.28);
    }}

    .pill-bonus {{
        background: rgba(168, 85, 247, 0.12);
        color: #A855F7;
        border: 1px solid rgba(168, 85, 247, 0.28);
    }}

    /* Bullet Diagnostics */
    .bullet-box {{
        background: {bg_surface_subtle};
        border-left: 4px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }}

    .bullet-box.strong {{
        border-left-color: #10B981;
        background: rgba(16, 185, 129, 0.06);
    }}

    .bullet-box.moderate {{
        border-left-color: #F59E0B;
        background: rgba(245, 158, 11, 0.06);
    }}

    .bullet-box.weak {{
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.06);
    }}

    /* Streamlit Element Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {bg_surface_subtle};
        padding: 4px;
        border-radius: 12px;
        border: 1px solid {border_subtle};
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.9rem;
        color: {text_secondary};
    }}

    .stTabs [aria-selected="true"] {{
        background: {bg_surface} !important;
        color: {text_accent} !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }}

    [data-testid="stSidebar"] {{
        background-color: {bg_surface};
        border-right: 1px solid {border_subtle};
    }}

    .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
    <div style="background: linear-gradient(135deg, #6366F1, #EC4899); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; color: white;">⚡</div>
    <div>
        <div style="font-weight: 800; font-size: 1.1rem; line-height: 1.1;">ResumeIntelligence</div>
        <div style="font-size: 0.75rem; color: #94A3B8;">AI/ML ATS Analytics v3.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation",
    [
        "🎯 AI/ML Resume Scorer",
        "⚖️ Dual Resume A/B Tester",
        "📝 AI Bullet Point Optimizer",
        "📚 AI/ML Skill Roadmap",
        "ℹ️ Scoring Methodology"
    ],
    index=0
)

st.sidebar.markdown("---")
theme_mode = st.sidebar.selectbox("🎨 UI Theme", ["Dark Mode (Glassmorphism)", "Light Mode (Clean Modern)"])
is_dark = "Dark" in theme_mode
inject_custom_css(dark_mode=is_dark)

st.sidebar.markdown("""
<div style="padding: 12px; border-radius: 10px; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); font-size: 0.78rem; color: #94A3B8; margin-top: 1.5rem;">
    <b style="color:#818CF8;">💡 Tip:</b> Use the <b>Sample Resumes</b> dropdown on the main page for instant 1-click test evaluations.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Plotly Radar Chart
# ---------------------------------------------------------
def render_radar_chart(radar_dims: dict, is_dark: bool = True):
    categories = list(radar_dims.keys())
    candidate_values = list(radar_dims.values())
    
    categories_plot = categories + [categories[0]]
    values_plot = candidate_values + [candidate_values[0]]
    benchmark_plot = [85, 80, 80, 75, 85, 85]

    fig = go.Figure()

    # Benchmark polygon
    fig.add_trace(go.Scatterpolar(
        r=benchmark_plot,
        theta=categories_plot,
        fill='toself',
        name='Target Benchmark',
        line=dict(color='rgba(148, 163, 184, 0.5)', width=1.5, dash='dash'),
        fillcolor='rgba(148, 163, 184, 0.06)'
    ))

    # Candidate polygon
    fig.add_trace(go.Scatterpolar(
        r=values_plot,
        theta=categories_plot,
        fill='toself',
        name='Candidate Profile',
        line=dict(color='#6366F1', width=2.8),
        fillcolor='rgba(99, 102, 241, 0.25)'
    ))

    grid_color = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.08)"
    text_color = "#F8FAFC" if is_dark else "#0F172A"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=text_color, size=9),
                gridcolor=grid_color
            ),
            angularaxis=dict(
                tickfont=dict(color=text_color, size=11, family="Plus Jakarta Sans", weight=600),
                gridcolor=grid_color
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color=text_color, size=11, family="Plus Jakarta Sans")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=35, r=35, t=15, b=25),
        height=340
    )
    return fig


# ---------------------------------------------------------
# Plotly Donut Chart
# ---------------------------------------------------------
def render_skills_donut(by_category: dict, is_dark: bool = True):
    labels = []
    counts = []
    for cat, skills in by_category.items():
        if len(skills) > 0:
            labels.append(cat)
            counts.append(len(skills))

    if not counts:
        labels = ["No Recognized Skills"]
        counts = [1]

    colors_palette = ['#6366F1', '#A855F7', '#EC4899', '#10B981', '#3B82F6', '#F59E0B', '#06B6D4', '#8B5CF6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        hole=0.6,
        marker=dict(colors=colors_palette),
        textinfo='percent',
        hoverinfo='label+value+percent'
    )])

    text_color = "#F8FAFC" if is_dark else "#0F172A"
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            xanchor="left",
            x=1.02,
            y=0.5,
            font=dict(color=text_color, size=10, family="Plus Jakarta Sans")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    return fig


# =========================================================
# PAGE 1: 🎯 AI/ML RESUME SCORER
# =========================================================
if nav_page == "🎯 AI/ML Resume Scorer":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Next-Gen NLP & TF-IDF Semantic Analytics</div>
        <div class="hero-title">AI/ML Resume Scorer & ATS Intelligence</div>
        <div class="hero-desc">Evaluate your resume with precision phrase extraction, semantic role alignment, X-Y-Z quantifiable impact scoring, and ATS compliance diagnostics.</div>
    </div>
    """, unsafe_allow_html=True)

    c_bench, c_up = st.columns([1, 1], gap="medium")

    with c_bench:
        st.markdown("##### 🎯 1. Target Job Benchmark")
        eval_mode = st.radio(
            "Target Type:",
            ["Preset AI/ML Role Benchmark", "Custom Job Description"],
            horizontal=True,
            label_visibility="collapsed"
        )

        selected_role = None
        custom_jd_content = ""

        if eval_mode == "Preset AI/ML Role Benchmark":
            selected_role = st.selectbox(
                "Select Target AI/ML Role",
                list(ROLE_BENCHMARKS.keys()),
                index=0
            )
            role_meta = ROLE_BENCHMARKS[selected_role]
            st.caption(f"📌 **Overview:** {role_meta['description']}")
            with st.expander("🔍 View Benchmark Core Competencies"):
                st.write("**Core Skills:** " + ", ".join(role_meta["core_skills"]))
                st.write("**Secondary Skills:** " + ", ".join(role_meta["secondary_skills"]))
        else:
            custom_jd_content = st.text_area(
                "Paste Custom Job Description",
                height=160,
                placeholder="Paste job posting details here..."
            )

    with c_up:
        st.markdown("##### 📤 2. Candidate Resume")
        sample_choice = st.selectbox(
            "⚡ Quick Test with Sample AI/ML Resumes:",
            ["(Upload My Own File)"] + list(SAMPLE_RESUMES.keys())
        )

        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            help="Upload your resume to perform deep multi-dimensional analytics."
        )

    # Resolve text
    resume_text = ""
    if sample_choice != "(Upload My Own File)":
        resume_text = SAMPLE_RESUMES[sample_choice]
        st.success(f"✓ Loaded sample: **{sample_choice}**")
    elif uploaded_file is not None:
        with st.spinner("Extracting resume content..."):
            resume_text = extract_text_from_file(uploaded_file)
            if not resume_text:
                st.error("Unable to extract text from the file. Please ensure it is not a scanned image PDF.")

    # ---------------------------------------------------------
    # ANALYTICS RESULTS
    # ---------------------------------------------------------
    if resume_text.strip():
        with st.spinner("Analyzing resume against benchmark with TF-IDF, SpaCy NLP, and ATS diagnostics..."):
            results = score_resume(
                resume_text=resume_text,
                target_role_name=selected_role,
                custom_jd_text=custom_jd_content if eval_mode != "Preset AI/ML Role Benchmark" else None
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        score = results["composite_score"]
        verdict = results["verdict"]
        verdict_color = results["verdict_color"]
        breakdown = results["scores_breakdown"]

        # HERO SCORE CARD
        st.markdown(f"""
        <div class="card-clean" style="border-left: 6px solid {verdict_color};">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 20px;">
                <div style="display: flex; align-items: center; gap: 24px;">
                    <div style="background: {verdict_color}18; border: 2px solid {verdict_color}; width: 96px; height: 96px; border-radius: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 800; color: {verdict_color}; line-height: 1;">{score}%</span>
                        <span style="font-size: 0.68rem; text-transform: uppercase; font-weight: 700; color: {verdict_color};">ATS Match</span>
                    </div>
                    <div>
                        <div style="font-size: 1.35rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; color: {verdict_color};">{verdict}</div>
                        <div style="font-size: 0.92rem; color: #94A3B8; max-width: 620px; margin-top: 4px;">{results['verdict_desc']}</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.04em;">Target Benchmark</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #818CF8;">{results['target_role']}</div>
                    <div style="font-size: 0.82rem; color: #10B981; margin-top: 4px;">✓ {results['skills_summary']['total_candidate_skills']} AI/ML Skills Extracted</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5 SUB-METRICS ROW
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-subcard">
                <div class="metric-subcard-title">🎯 Hard Skills</div>
                <div class="metric-subcard-value">{breakdown['hard_skill_score']}%</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Core Match Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-subcard">
                <div class="metric-subcard-title">📐 Semantic Fit</div>
                <div class="metric-subcard-value">{breakdown['semantic_score']}%</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">TF-IDF Cosine</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-subcard">
                <div class="metric-subcard-title">💥 Quant Impact</div>
                <div class="metric-subcard-value">{breakdown['impact_score']}%</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">X-Y-Z Density</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-subcard">
                <div class="metric-subcard-title">⚙️ MLOps Readiness</div>
                <div class="metric-subcard-value">{breakdown['mlops_score']}%</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Production Tools</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-subcard">
                <div class="metric-subcard-title">🛡️ ATS Health</div>
                <div class="metric-subcard-value">{breakdown['ats_health_score']}%</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Format & Structure</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # TABBED SECTIONS
        tab_radar, tab_skills, tab_impact, tab_ats, tab_view, tab_export = st.tabs([
            "📊 5D Competency & Visuals",
            "🧠 Skill Gap & Learning Plan",
            "🔍 Bullet-Point & Impact Audit",
            "📋 ATS Health & Readability",
            "📄 Parsed Content Viewer",
            "📥 Download Audit Report"
        ])

        # TAB 1: RADAR & VISUALS
        with tab_radar:
            col_rad, col_don = st.columns([1.1, 0.9], gap="large")
            with col_rad:
                st.markdown("##### 📡 5-Dimensional Competency Radar")
                st.caption("Visualizes candidate proficiency across 5 key dimensions vs target benchmark.")
                fig_radar = render_radar_chart(results["radar_dimensions"], is_dark=is_dark)
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_don:
                st.markdown("##### 🍩 Candidate Skill Taxonomy Distribution")
                st.caption("Distribution of recognized skills across specialized AI/ML domains.")
                fig_donut = render_skills_donut(results["skills_summary"]["by_category"], is_dark=is_dark)
                st.plotly_chart(fig_donut, use_container_width=True)

        # TAB 2: SKILL GAP & LEARNING PLAN
        with tab_skills:
            # Quick skill search
            search_query = st.text_input("🔍 Filter Candidate Skills:", placeholder="Type to filter extracted skills (e.g. PyTorch, Docker, RAG)...").lower()
            
            col_m, col_g = st.columns(2, gap="large")
            with col_m:
                st.markdown("##### ✅ Matched Core Skills")
                matched_core = results["skills_summary"]["matched_core"]
                if search_query:
                    matched_core = [s for s in matched_core if search_query in s]

                if matched_core:
                    html_chips = "".join([f'<span class="pill pill-matched">✓ {s}</span>' for s in matched_core])
                    st.markdown(html_chips, unsafe_allow_html=True)
                else:
                    st.warning("No core required skills matched your filter.")

                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("##### ⚡ Matched Secondary Skills")
                matched_sec = results["skills_summary"]["matched_secondary"]
                if search_query:
                    matched_sec = [s for s in matched_sec if search_query in s]

                if matched_sec:
                    html_sec = "".join([f'<span class="pill pill-secondary">⚡ {s}</span>' for s in matched_sec])
                    st.markdown(html_sec, unsafe_allow_html=True)
                else:
                    st.caption("No secondary skills matched.")

                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("##### ➕ Bonus & Distinct Competencies")
                bonus_s = results["skills_summary"]["bonus_skills"]
                if search_query:
                    bonus_s = [s for s in bonus_s if search_query in s]

                if bonus_s:
                    html_bonus = "".join([f'<span class="pill pill-bonus">★ {s}</span>' for s in bonus_s])
                    st.markdown(html_bonus, unsafe_allow_html=True)

            with col_g:
                st.markdown("##### ❌ Missing Critical Skills (Add to Resume)")
                missing_core = results["skills_summary"]["missing_core"]
                if missing_core:
                    html_missing = "".join([f'<span class="pill pill-missing">✕ {s}</span>' for s in missing_core])
                    st.markdown(html_missing, unsafe_allow_html=True)
                else:
                    st.success("🎉 Perfect match! No core skills missing.")

                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("##### 📚 Recommended Learning Roadmap")
                st.caption("Curated resources to bridge detected skill gaps:")
                for rec in results["learning_recommendations"][:6]:
                    st.markdown(f"- **[{rec['priority'].split()[0]}] [{rec['skill'].title()}]({rec['link']})**: *{rec['title']}*")

        # TAB 3: RECRUITER BULLET-POINT AUDIT
        with tab_impact:
            impact = results["impact_analysis"]
            
            c_v, c_met, c_st = st.columns(3)
            with c_v:
                st.markdown(f"**Action Verbs Found ({len(impact['strong_verbs_found'])})**")
                if impact['strong_verbs_found']:
                    st.write(", ".join(impact['strong_verbs_found'][:15]))
                else:
                    st.warning("No strong action verbs detected.")
            with c_met:
                st.markdown(f"**Quantified Metrics ({len(impact['metrics_found'])})**")
                if impact['metrics_found']:
                    st.write(", ".join(impact['metrics_found']))
                else:
                    st.warning("No quantifiable metrics (%, $, latency, scale) detected.")
            with c_st:
                st.markdown("**Bullet Quality Scorecard**")
                st.write(f"- Analyzed Bullets: **{impact['total_bullets_analyzed']}**")
                st.write(f"- High-Impact Bullets: **{impact['strong_bullets_count']}**")
                if impact['weak_phrases_found']:
                    st.error(f"⚠️ Weak phrases: {', '.join(impact['weak_phrases_found'])}")
                else:
                    st.success("✓ No weak passive phrases detected!")

            st.markdown("---")
            st.markdown("##### 📝 Line-by-Line Bullet Point Diagnostics & Rewrites")
            for item in impact["bullet_analysis"][:10]:
                box_class = "strong" if "Strong" in item["rating"] else ("moderate" if "Moderate" in item["rating"] else "weak")
                st.markdown(f"""
                <div class="bullet-box {box_class}">
                    <div style="font-size:0.85rem; font-weight:700; margin-bottom:2px;">{item['rating']}</div>
                    <div style="font-size:0.92rem; margin-bottom:4px;"><i>"{item['bullet']}"</i></div>
                    <div style="font-size:0.84rem; color:#94A3B8;">💡 <b>Suggestion:</b> {item['suggestion']}</div>
                </div>
                """, unsafe_allow_html=True)

        # TAB 4: ATS HEALTH & READABILITY
        with tab_ats:
            readability = results["readability"]
            sec_eval = results["section_evaluation"]
            contact = results["contact_info"]

            c_c1, c_c2 = st.columns(2, gap="large")
            with c_c1:
                st.markdown("##### 👤 Contact Channels")
                st.write(f"- Email: {'✅ ' + contact['email'] if contact['email'] else '❌ Missing'}")
                st.write(f"- Phone: {'✅ ' + contact['phone'] if contact['phone'] else '❌ Missing'}")
                st.write(f"- LinkedIn: {'✅ ' + contact['linkedin'] if contact['linkedin'] else '❌ Missing'}")
                st.write(f"- GitHub: {'✅ ' + contact['github'] if contact['github'] else '❌ Missing'}")
                st.write(f"- Portfolio: {'✅ ' + contact['portfolio'] if contact['portfolio'] else '❌ Missing'}")

                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("##### 📑 Section Completeness")
                st.write(f"- Found Sections: **{', '.join(sec_eval['found_sections'])}**")
                if sec_eval["missing_critical"]:
                    st.error(f"Missing Essential Sections: {', '.join(sec_eval['missing_critical'])}")
                else:
                    st.success("✓ All essential sections present!")

            with c_c2:
                st.markdown("##### 📖 Document Readability")
                st.metric("Word Count", f"{readability['word_count']} words", readability['length_status'])
                st.metric("Flesch Reading Ease", f"{readability['flesch_score']}", readability['readability_status'])
                st.info("💡 **ATS Tip:** Maintain 450–900 words with clean standard section headers for 100% ATS parser compatibility.")

        # TAB 5: PARSED CONTENT
        with tab_view:
            st.markdown("##### 📄 Parsed Sections")
            for sec_name, sec_content in results["sections"].items():
                with st.expander(f"📌 Section: {sec_name} ({len(sec_content.splitlines())} lines)"):
                    st.text(sec_content)

        # TAB 6: EXPORT REPORT
        with tab_export:
            st.markdown("##### 📥 Export Candidate Audit Report")
            pdf_bytes = generate_pdf_report(results)
            md_text = generate_markdown_report(results)

            c_p, c_m = st.columns(2)
            with c_p:
                st.download_button(
                    label="📄 Download PDF Audit Report",
                    data=pdf_bytes,
                    file_name="AI_ML_Resume_Audit_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with c_m:
                st.download_button(
                    label="📝 Download Markdown Summary",
                    data=md_text,
                    file_name="AI_ML_Resume_Audit_Report.md",
                    mime="text/markdown",
                    use_container_width=True
                )


# =========================================================
# PAGE 2: ⚖️ DUAL RESUME A/B TESTER
# =========================================================
elif nav_page == "⚖️ Dual Resume A/B Tester":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚖️ Comparative Candidate Analytics</div>
        <div class="hero-title">Dual Resume A/B Match Tester</div>
        <div class="hero-desc">Compare two resumes side-by-side against a benchmark to identify the strongest candidate or optimize your resume iterations.</div>
    </div>
    """, unsafe_allow_html=True)

    target_role_ab = st.selectbox("🎯 Select Target Role Benchmark", list(ROLE_BENCHMARKS.keys()), index=0)

    c_r1, c_r2 = st.columns(2, gap="medium")
    with c_r1:
        st.markdown("##### 📄 Resume Version A")
        s_choice_a = st.selectbox("Sample A:", ["(Upload File)"] + list(SAMPLE_RESUMES.keys()), index=1)
        up_a = st.file_uploader("Upload Resume A", type=["pdf", "docx", "txt"], key="up_a")
        text_a = SAMPLE_RESUMES[s_choice_a] if s_choice_a != "(Upload File)" else (extract_text_from_file(up_a) if up_a else "")

    with c_r2:
        st.markdown("##### 📄 Resume Version B")
        s_choice_b = st.selectbox("Sample B:", ["(Upload File)"] + list(SAMPLE_RESUMES.keys()), index=2)
        up_b = st.file_uploader("Upload Resume B", type=["pdf", "docx", "txt"], key="up_b")
        text_b = SAMPLE_RESUMES[s_choice_b] if s_choice_b != "(Upload File)" else (extract_text_from_file(up_b) if up_b else "")

    if text_a and text_b:
        with st.spinner("Analyzing and comparing both candidate resumes..."):
            res_a = score_resume(text_a, target_role_name=target_role_ab)
            res_b = score_resume(text_b, target_role_name=target_role_ab)

        st.markdown("---")
        diff = res_a["composite_score"] - res_b["composite_score"]
        if diff > 0:
            st.success(f"🏆 **Resume Version A leads by +{diff}% overall match score!**")
        elif diff < 0:
            st.success(f"🏆 **Resume Version B leads by +{abs(diff)}% overall match score!**")
        else:
            st.info("🤝 Both resumes are evenly matched.")

        # Comparison Radar Overlay
        st.markdown("##### 📡 Comparative 5D Radar Overlay")
        fig_ab = go.Figure()
        categories = list(res_a["radar_dimensions"].keys())
        categories_plot = categories + [categories[0]]

        fig_ab.add_trace(go.Scatterpolar(
            r=list(res_a["radar_dimensions"].values()) + [list(res_a["radar_dimensions"].values())[0]],
            theta=categories_plot,
            fill='toself',
            name='Resume Version A',
            line=dict(color='#6366F1', width=2.5),
            fillcolor='rgba(99, 102, 241, 0.2)'
        ))

        fig_ab.add_trace(go.Scatterpolar(
            r=list(res_b["radar_dimensions"].values()) + [list(res_b["radar_dimensions"].values())[0]],
            theta=categories_plot,
            fill='toself',
            name='Resume Version B',
            line=dict(color='#EC4899', width=2.5),
            fillcolor='rgba(236, 72, 153, 0.2)'
        ))

        text_color = "#F8FAFC" if is_dark else "#0F172A"
        grid_color = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.08)"
        fig_ab.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=text_color, size=9), gridcolor=grid_color),
                angularaxis=dict(tickfont=dict(color=text_color, size=11, family="Plus Jakarta Sans", weight=600), gridcolor=grid_color),
                bgcolor="rgba(0,0,0,0)"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color=text_color)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_ab, use_container_width=True)

        # Comparison Data Table
        comp_df = pd.DataFrame({
            "Metric": ["Overall Composite Score", "Hard Skills Match", "TF-IDF Semantic Fit", "Quantifiable Impact", "MLOps & Deployment", "ATS Health", "Total Skills Extracted", "Action Verbs"],
            "Resume Version A": [f"{res_a['composite_score']}%", f"{res_a['scores_breakdown']['hard_skill_score']}%", f"{res_a['scores_breakdown']['semantic_score']}%", f"{res_a['scores_breakdown']['impact_score']}%", f"{res_a['scores_breakdown']['mlops_score']}%", f"{res_a['scores_breakdown']['ats_health_score']}%", res_a['skills_summary']['total_candidate_skills'], len(res_a['impact_analysis']['strong_verbs_found'])],
            "Resume Version B": [f"{res_b['composite_score']}%", f"{res_b['scores_breakdown']['hard_skill_score']}%", f"{res_b['scores_breakdown']['semantic_score']}%", f"{res_b['scores_breakdown']['impact_score']}%", f"{res_b['scores_breakdown']['mlops_score']}%", f"{res_b['scores_breakdown']['ats_health_score']}%", res_b['skills_summary']['total_candidate_skills'], len(res_b['impact_analysis']['strong_verbs_found'])]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# =========================================================
# PAGE 3: 📝 AI BULLET POINT OPTIMIZER
# =========================================================
elif nav_page == "📝 AI Bullet Point Optimizer":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">📝 Google X-Y-Z Transformer</div>
        <div class="hero-title">AI Bullet Point & Impact Optimizer</div>
        <div class="hero-desc">Transform passive, vague resume bullets into high-impact, metric-driven accomplishments following Google's X-Y-Z formula.</div>
    </div>
    """, unsafe_allow_html=True)

    sample_bullets = [
        "Worked on LLM prompt engineering and helped team deploy RAG application.",
        "Responsible for building customer churn prediction model with Python and random forest.",
        "Architected enterprise RAG system with LlamaIndex and Pinecone, reducing query latency by 45% for 500k daily users."
    ]

    chosen_sample = st.selectbox("Choose an example bullet to test or write your own:", ["(Write my own)"] + sample_bullets)
    default_text = "" if chosen_sample == "(Write my own)" else chosen_sample

    user_bullet = st.text_area("Paste or write a resume bullet point:", value=default_text, height=90)

    if user_bullet.strip():
        b_lower = user_bullet.lower()
        has_verb = any(v in b_lower for v in STRONG_ACTION_VERBS)
        has_weak = any(w in b_lower for w in WEAK_PASSIVE_PHRASES)
        has_metric = bool(re.search(r'(\d+%|\$\d+|\d+x|\d+\s?ms|\d+\s?k|\d+\s?m)', user_bullet, re.IGNORECASE))

        st.markdown("##### 📊 Live Diagnostic")
        c1, c2, c3 = st.columns(3)
        with c1:
            if has_verb: st.success("✅ Strong Action Verb")
            else: st.error("❌ Missing Action Verb")
        with c2:
            if has_metric: st.success("✅ Quantifiable Metric")
            else: st.error("❌ Missing Metric (%, $, scale)")
        with c3:
            if not has_weak: st.success("✅ Clean (No passive voice)")
            else: st.warning("⚠️ Passive phrase detected")

        st.markdown("---")
        st.markdown("##### 💡 3-Tier Rewrite Suggestions (Junior → Senior Staff)")
        st.markdown("""
        - 🔹 **Tier 1 (Metric Enhanced):** *"Engineered customer churn model using Scikit-Learn Random Forest, improving prediction accuracy by 18% across 200k user accounts."*
        - ⚡ **Tier 2 (Production Scale):** *"Architected automated churn prediction pipeline using XGBoost and FastAPI on Docker, reducing quarterly subscriber loss by $120k."*
        - 🏆 **Tier 3 (Staff/Lead X-Y-Z):** *"Spearheaded end-to-end predictive intelligence system with MLflow and AWS SageMaker, decreasing customer churn by 22% ($450k ARR impact) with sub-50ms real-time inference."*
        """)


# =========================================================
# PAGE 4: 📚 AI/ML SKILL TAXONOMY ROADMAP
# =========================================================
elif nav_page == "📚 AI/ML Skill Roadmap":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">📚 2025/2026 Skill Taxonomy</div>
        <div class="hero-title">AI/ML Competency Taxonomy & Roadmap</div>
        <div class="hero-desc">Explore industry-standard skills, frameworks, and curated learning paths for each specialized AI/ML domain.</div>
    </div>
    """, unsafe_allow_html=True)

    for cat_name, skills_list in CATEGORIZED_SKILLS.items():
        with st.expander(f"🔹 {cat_name} ({len(skills_list)} Skills)"):
            html_chips = "".join([f'<span class="pill pill-secondary">{s}</span>' for s in skills_list])
            st.markdown(html_chips, unsafe_allow_html=True)


# =========================================================
# PAGE 5: ℹ️ SCORING METHODOLOGY
# =========================================================
elif nav_page == "ℹ️ Scoring Methodology":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🔬 Algorithmic Transparency</div>
        <div class="hero-title">Scoring Methodology & Rubric</div>
        <div class="hero-desc">Understand how our NLP pipeline, TF-IDF vectorizer, and composite scoring metrics evaluate candidate profiles.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Multi-Faceted Scoring Architecture

    Our scoring algorithm employs a multi-dimensional weighted composite model:

    1. **Hard Skill Competency Match (35%)**:
       - Extracts exact technical phrases via SpaCy `PhraseMatcher` and boundary-aware regex.
       - Accurately captures multi-word skills like *"Large Language Models"*, *"Retrieval Augmented Generation"*, *"Prompt Engineering"*, *"Diffusion Models"*, and *"Vision Transformers"*.

    2. **TF-IDF Semantic Cosine Similarity (20%)**:
       - Vectorizes resume and target job benchmarks using sublinear TF-IDF $(1,2)$-grams.
       - Computes mathematical cosine similarity in high-dimensional semantic space.

    3. **Quantifiable Impact & Action Verb Strength (15%)**:
       - Evaluates dense presence of industry action verbs (*Architected, Deployed, Benchmarked, Optimized*).
       - Detects quantifiable outcome metrics ($%, \$, \text{latency, throughput, scale}$).

    4. **MLOps & Production Readiness (15%)**:
       - Checks for end-to-end operational competency: Docker, Kubernetes, CI/CD, MLflow, FastAPI, Triton.

    5. **ATS Formatting & Readability Health (15%)**:
       - Evaluates standard section segmentation, contact completeness (email, phone, LinkedIn/GitHub), and Flesch Reading Ease.
    """)
