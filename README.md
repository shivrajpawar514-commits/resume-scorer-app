# ⚡ AI/ML Resume Intelligence & ATS Scorer

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SpaCy](https://img.shields.io/badge/SpaCy-3.8%2B-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**An advanced, production-grade AI/ML Resume Scorer & ATS Analytics Engine powered by multi-tier NLP parsing, TF-IDF semantic embeddings, Google X-Y-Z quantifiable impact diagnostics, and interactive 5D radar visualizations.**

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

### 3. 📊 Interactive Visualizations
- **5D Competency Radar (Plotly)**: Visualizes candidate profile vs ideal benchmark across 5 core dimensions.
- **Skill Taxonomy Donut Chart**: Breaks down extracted technical skills across specialized AI/ML domains.
- **Live Search & Filter**: Real-time filterable skill matrix with color-coded badges (`✓ Matched`, `✕ Missing`, `⚡ Secondary`, `★ Bonus`).

### 4. 🔍 Recruiter Bullet-Point & Impact Audit
- Evaluates compliance with Google's X-Y-Z formula (*"Accomplished [X] as measured by [Y], by doing [Z]"*).
- Line-by-line ratings (Strong, Moderate, Needs Improvement) with AI suggestions for rewriting weak bullet points.

### 5. ⚖️ Dual Resume A/B Match Tester
- Side-by-side comparison of two resumes with winning delta badges and comparative 5D radar overlays.

### 6. 📝 Interactive AI Bullet Optimizer
- Test and transform passive resume bullet points into high-impact 3-tier statements (Junior → Mid → Senior Staff level).

### 7. 📥 Exportable Audit Reports
- Download formatted **PDF Audit Reports** (built with ReportLab) or **Markdown Summaries**.

---

## 🏗️ Project Architecture

```
ResumeScorerApp/
├── app.py                      # Main Streamlit Web Application (Glassmorphic UI)
├── requirements.txt            # Project dependencies
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

### 2. Set Up Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

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
