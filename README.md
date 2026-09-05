# ⚡ AI/ML Resume Scorer & ATS Intelligence Platform

A next-generation **AI/ML Resume Scorer and Advanced Analytics Platform** built with Python, Streamlit, SpaCy NLP, and Scikit-Learn TF-IDF semantic vector embeddings. Evaluates candidate resumes against modern 2025/2026 AI/ML engineering benchmarks, computes 5D competency scores, detects quantifiable impact metrics, and performs deep ATS compliance audits.

---

## 🌟 Key Features

- **Multi-Faceted 5D Scoring Engine (0–100%)**:
  - 🎯 **Hard Skills Match (35%)**: Precision phrase extraction using SpaCy `PhraseMatcher` (supports multi-word skills like *"Large Language Models"*, *"Retrieval Augmented Generation"*, *"Prompt Engineering"*, *"Diffusion Models"*).
  - 📐 **TF-IDF Semantic Cosine Similarity (20%)**: Mathematical contextual vector similarity against target role profiles or custom Job Descriptions.
  - 💥 **Quantifiable Impact & Action Verbs (15%)**: Detection of action verbs (*Architected, Deployed, Benchmarked*) and measurable metrics (`%`, `$`, scale numbers, latency).
  - ⚙️ **MLOps & Production Readiness (15%)**: Assesses deployment readiness in Docker, Kubernetes, CI/CD, MLflow, FastAPI, and Triton.
  - 🛡️ **ATS Formatting & Readability Health (15%)**: Evaluates section structure completeness, contact channels, and Flesch Reading Ease.

- **8 Specialized AI/ML Track Benchmarks**:
  - 🤖 Generative AI / LLM Engineer
  - 🧠 Machine Learning Engineer
  - ⚙️ MLOps Engineer
  - 🔬 Data Scientist
  - 👁️ Computer Vision Engineer
  - 🗣️ NLP / Speech Engineer
  - 📊 Data Analyst / BI Specialist
  - 🌐 Full-Stack AI Engineer

- **Custom Job Description (JD) Mode**: Paste any custom job description to run a bespoke gap analysis and match score.
- **Interactive 5D Competency Radar (Plotly)**: Visualizes candidate dimensions against industry benchmarks.
- **Dual Resume A/B Tester**: Side-by-side comparison of two resumes with a comparative radar overlay.
- **AI Bullet Point Optimizer (X-Y-Z Transformer)**: Transforms weak, passive bullet points into impactful accomplishments following Google's X-Y-Z formula.
- **Downloadable PDF & Markdown Reports**: Instant export of candidate audit reports via ReportLab.

---

## 📂 Project Architecture

```
ResumeScorerApp/
├── app.py                      # Main Streamlit Application with modern UI
├── requirements.txt            # Python dependencies
├── job_keywords.json           # Reference keyword mapping
├── analyzer/
│   ├── __init__.py
│   ├── extractor.py            # Multi-format document parser (PDF, DOCX, TXT)
│   ├── section_parser.py       # Smart section chunking & completeness scoring
│   ├── skills_taxonomy.py      # AI/ML skill taxonomy & 8 role benchmarks
│   ├── matcher.py              # SpaCy PhraseMatcher & regex phrase extractor
│   ├── impact_analyzer.py      # Action verb, metric density & X-Y-Z evaluator
│   ├── scoring_engine.py       # 5D composite scoring & radar dimensions
│   ├── report_builder.py       # PDF & Markdown report generator
│   └── sample_resumes.py       # Preloaded sample AI/ML resumes for 1-click testing
```

---

## 🚀 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/shivrajpawar514-commits/resume-scorer-app.git
cd resume-scorer-app
```

### 2. Set up virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies & SpaCy language model
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 License
MIT License
