import streamlit as st
import PyPDF2
import spacy
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------
# CSS STYLES FOR MODERN UI + DARK MODE
# ---------------------------------------
def apply_dark_mode():
    dark_css = """
    <style>
    body, .stApp {
        background-color: #0e1117 !important;
        color: white !important;
    }
    .stTextInput textarea, .stTextArea textarea, .stSelectbox, .stMarkdown, .stDataFrame {
        background-color: #161a23 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #4a4f57 !important;
        color: white !important;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

def apply_light_mode():
    light_css = """
    <style>
    body, .stApp {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)

# ---------------------------------------
# Predefined Skill Sets
# ---------------------------------------
ROLE_SKILLS = {
    "AI/ML Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "data preprocessing", "linear regression", "classification", "nlp",
        "model training", "random forest"
    ],

    "UI/UX Designer": [
        "figma", "adobe xd", "wireframing", "prototyping", "user research",
        "ui design", "ux design", "graphic design", "photoshop"
    ],

    "Data Analyst": [
        "excel", "sql", "power bi", "tableau", "data visualization",
        "python", "statistics", "pandas", "data cleaning"
    ],

    "Web Developer": [
        "html", "css", "javascript", "react", "nodejs", "express",
        "bootstrap", "mongodb", "frontend", "backend"
    ],

    "Cloud Engineer": [
        "aws", "azure", "gcp", "docker", "kubernetes",
        "devops", "terraform", "cloud security", "linux"
    ]
}

# ---------------------------------------
# Sidebar
# ---------------------------------------
st.sidebar.title("⚙ App Settings")

# Navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📄 Resume Analysis", "ℹ About"])

# Theme selection
theme = st.sidebar.selectbox("Theme Mode", ["Light", "Dark"])

# Apply theme
if theme == "Dark":
    apply_dark_mode()
else:
    apply_light_mode()


# ----------------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------------
if page == "🏠 Home":
    st.title("✨ Resume Analyzer App")
    st.write("A modern AI-powered Resume Scoring & Recommendation Tool.")


# ----------------------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------------------
elif page == "ℹ About":
    st.title("ℹ About This App")
    st.write("""
    This app extracts resume text, identifies important skills, and evaluates suitability for various roles using NLP.
    """)


# ----------------------------------------------------------------
# RESUME ANALYSIS PAGE
# ----------------------------------------------------------------
elif page == "📄 Resume Analysis":

    st.markdown("<h1 style='text-align:center;'>📄 Resume Analysis</h1>", unsafe_allow_html=True)

    role = st.selectbox("🎯 Select Job Role", list(ROLE_SKILLS.keys()))

    uploaded_file = st.file_uploader("📤 Upload your resume (PDF)", type="pdf")

    if uploaded_file:
        # Extract Text
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()

        st.subheader("📌 Extracted Resume Text")
        st.text_area("", text, height=200)

        doc = nlp(text.lower())
        tokens = [t.text for t in doc if t.is_alpha]

        required_skills = ROLE_SKILLS[role]
        matched = [s for s in required_skills if s in tokens]
        missing = [s for s in required_skills if s not in tokens]

        score = int((len(matched) / len(required_skills)) * 100)

        # Animated Progress Bar
        st.subheader("📊 Resume Score")
        st.progress(score / 100)
        st.write(f"### ⭐ Score: **{score}%**")

        # Pie Chart
        labels = ["Matched", "Missing"]
        sizes = [len(matched), len(missing)]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        # Skill Chips UI
        st.subheader("✅ Matched Skills")
        for skill in matched:
            st.markdown(f"<span style='padding:6px 10px; background:#d1ffd1; border-radius:12px; margin:4px; display:inline-block;'>{skill}</span>", unsafe_allow_html=True)

        st.subheader("❌ Missing Skills")
        for skill in missing:
            st.markdown(f"<span style='padding:6px 10px; background:#ffd1d1; border-radius:12px; margin:4px; display:inline-block;'>{skill}</span>", unsafe_allow_html=True)

        # Suitability
        st.subheader("🎯 Final Suitability")
        if score >= 70:
            st.success(f"You are suitable for the **{role}** role! 🎉")
        elif score >= 40:
            st.warning(f"Partially suitable. Improve missing skills.")
        else:
            st.error(f"Not suitable yet. Work on skill gaps.")

        # Recommendations
        if missing:
            st.subheader("📚 Skills to Learn")
            st.write(", ".join(missing))
