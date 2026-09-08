# ⚡ AI/ML Resume Intelligence & ATS Scorer

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SpaCy](https://img.shields.io/badge/SpaCy-3.8%2B-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CI Status](https://img.shields.io/badge/CI-Passing-10B981?style=for-the-badge&logo=github-actions&logoColor=white)

**An advanced, production-grade AI/ML Resume Scorer, ATS Analytics Platform & REST API Service powered by multi-tier NLP parsing, TF-IDF semantic embeddings, Google X-Y-Z quantifiable impact diagnostics, and interactive 5D radar visualizations.**

</div>

---

## 🌟 Key Features

### 1. 🎯 Multi-Dimensional AI/ML Scoring (0–100%)
- **Hard Skill Competency (35%)**: Precision multi-word phrase matching via SpaCy `PhraseMatcher` (*"Large Language Models"*, *"Retrieval Augmented Generation"*, *"Prompt Engineering"*, *"Diffusion Models"*, *"Vision Transformers"*).
- **TF-IDF Semantic Cosine Similarity (20%)**: High-dimensional semantic vector alignment against role benchmarks or custom Job Descriptions.
- **Quantifiable Impact & Action Verbs (15%)**: Evaluates strong action verbs (*Architected, Deployed, Benchmarked*) and quantifiable outcome metrics (`%`, `$`, latency, scale).
- **MLOps & Production Readiness (15%)**: Assesses production deployment skills (Docker, Kubernetes, MLflow, CI/CD, FastAPI, Triton).
- **ATS Formatting & Readability (15%)**: Evaluates standard section completeness, contact channels, and Flesch Reading Ease.

### 2. 🤖 8 Specialized AI/ML Track Benchmarks + Custom JD Support
- 🤖 **Generative AI / LLM Engineer**
- 🧠 **Machine Learning Engineer**
- ⚙️ **MLOps Engineer**
- 🔬 **Data Scientist**
- 👁️ **Computer Vision Engineer**
- 🗣️ **NLP / Speech Engineer**
- 📊 **Data Analyst / BI Specialist**
- 🌐 **Full-Stack AI Engineer**
- 📋 **Custom Job Description Mode**: Paste or upload any custom JD text for instant bespoke alignment analysis.

### 3. 🌐 Production REST API (FastAPI & Swagger Docs)
- High-throughput headless REST API service (`api.py`) with Swagger UI (`/docs`) and ReDoc (`/redoc`).
- Single & batch scoring endpoints, multipart resume document uploads, PDF/Markdown report generation, and role taxonomy inspection.

### 4. 📊 Interactive Visualizations & Streamlit Web UI
- **5D Competency Radar (Plotly)**: Visualizes candidate profile vs ideal benchmark across 5 core dimensions.
- **Skill Taxonomy Donut Chart**: Breaks down extracted technical skills across specialized AI/ML domains.
- **Live Search & Filter**: Real-time filterable skill matrix with color-coded badges (`✓ Matched`, `✕ Missing`, `⚡ Secondary`, `★ Bonus`).

### 5. 🔍 Recruiter Bullet-Point & Impact Audit
- Evaluates compliance with Google's X-Y-Z formula (*"Accomplished [X] as measured by [Y], by doing [Z]"*).
- Line-by-line ratings (Strong, Moderate, Needs Improvement) with AI suggestions for rewriting weak bullet points.

### 6. 💻 CLI & Batch Evaluation Tool
- Command-line interface (`cli.py`) for automated scoring and bulk evaluations across entire directories of resumes with JSON, CSV leaderboard, and PDF export.

### 7. 🐳 Docker & CI/CD Ready
- Containerized Docker deployment (`Dockerfile`) and automated GitHub Actions test pipeline (`.github/workflows/ci.yml`).

---

## 🏗️ Project Architecture

```
ResumeScorerApp/
├── app.py                      # Main Streamlit Web Application (Glassmorphic UI)
├── api.py                      # Production FastAPI REST API Service
├── cli.py                      # CLI batch evaluation & ranking tool
├── pyproject.toml              # Build & tool configuration metadata
├── Dockerfile                  # Production container configuration
├── requirements.txt            # Project dependencies
├── .editorconfig               # Code formatting standard configuration
├── .streamlit/
│   └── config.toml             # Streamlit UI theme presets & server settings
├── .github/
│   └── workflows/ci.yml        # GitHub Actions CI automated testing
├── tests/
│   ├── test_analyzer.py        # Core NLP & scoring engine unit tests
│   ├── test_api.py             # FastAPI REST endpoint integration tests
│   └── test_cli.py             # CLI batch & CSV export unit tests
├── analyzer/
│   ├── __init__.py
│   ├── skills_taxonomy.py      # 2025/2026 AI/ML taxonomies & role benchmarks
│   ├── extractor.py            # PDF (PyMuPDF), DOCX & TXT text and contact extractor
│   ├── section_parser.py       # Smart section chunker (Experience, Projects, Education, etc.)
│   ├── matcher.py              # SpaCy PhraseMatcher + regex boundary skill extractor
│   ├── impact_analyzer.py      # Bullet-point & quantifiable metric evaluator
│   ├── scoring_engine.py       # 5D composite scoring algorithm & radar data builder
│   ├── report_builder.py       # ReportLab PDF & Markdown report generator
│   └── sample_resumes.py       # Preloaded realistic AI/ML resumes for 1-click testing
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shivrajpawar514-commits/resume-scorer-app.git
cd resume-scorer-app
```

### 2. Local Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the REST API Server
```bash
uvicorn api:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc UI: `http://localhost:8000/redoc`

#### Example REST API Request:
```bash
curl -X POST "http://localhost:8000/api/score" \
     -H "Content-Type: application/json" \
     -d '{
       "resume_text": "Experienced Machine Learning Engineer with PyTorch, Docker, Kubernetes, and LLM fine-tuning.",
       "target_role": "🤖 Generative AI / LLM Engineer"
     }'
```

### 5. Run via CLI
```bash
# Evaluate a single resume against a role
python cli.py --resume path/to/resume.pdf --role "🤖 Generative AI / LLM Engineer" --pdf-out audit.pdf

# Run batch evaluation across a folder of resumes with CSV leaderboard export
python cli.py --batch-dir ./resumes --role "🧠 Machine Learning Engineer" --csv-out leaderboard.csv --json-out batch_results.json
```

### 6. Run via Docker
```bash
# Build Docker image
docker build -t resume-scorer-app .

# Run container
docker run -p 8501:8501 -p 8000:8000 resume-scorer-app
```

### 7. Run Automated Tests
```bash
python -m unittest discover tests -v
```

---

## 📊 Evaluation Rubric

| Dimension | Weight | Description |
| :--- | :---: | :--- |
| **Core AI/ML Hard Skills** | **35%** | Phrase-level match against required technical skills. |
| **TF-IDF Semantic Fit** | **20%** | Mathematical cosine similarity between resume & JD. |
| **Quantifiable Impact** | **15%** | Metric density (`%`, `$`, scale) & strong action verbs. |
| **MLOps & Infrastructure** | **15%** | Production deployment, Docker, CI/CD, pipelines. |
| **ATS Health & Readability**| **15%** | Flesch reading score, section headers, contact info. |

---

## 📜 License
MIT License. Free for educational and commercial use.
