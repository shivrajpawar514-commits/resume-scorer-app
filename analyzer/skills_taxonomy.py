"""
AI/ML Skills Taxonomy & Role Benchmarks
Comprehensive taxonomy of modern AI/ML, Data Science, and Engineering skills (2025/2026).
"""

CATEGORIZED_SKILLS = {
    "Generative AI & LLMs": [
        "large language models", "llm", "rag", "retrieval augmented generation",
        "langchain", "llamaindex", "prompt engineering", "fine-tuning",
        "peft", "lora", "qlora", "vector databases", "pinecone", "chromadb",
        "faiss", "weaviate", "milvus", "qdrant", "huggingface", "transformers",
        "vllm", "ollama", "openai api", "anthropic api", "gemini api",
        "diffusion models", "stable diffusion", "midjourney", "rlhf", "dpo",
        "agentic workflows", "crewai", "autogen", "semantic search", "embeddings",
        "context window optimization", "guardrails", "triton inference server"
    ],
    "Deep Learning & AI Architectures": [
        "deep learning", "pytorch", "tensorflow", "keras", "neural networks",
        "cnn", "convolutional neural networks", "rnn", "recurrent neural networks",
        "lstm", "transformer models", "bert", "gpt", "vision transformers",
        "computer vision", "opencv", "yolo", "object detection", "image segmentation",
        "nlp", "natural language processing", "named entity recognition", "ner",
        "sentiment analysis", "spacy", "nltk", "speech recognition", "whisper",
        "generative adversarial networks", "gans", "autoencoders", "jax", "flax"
    ],
    "Classical Machine Learning & Modeling": [
        "machine learning", "scikit-learn", "xgboost", "lightgbm", "catboost",
        "random forest", "decision trees", "gradient boosting", "linear regression",
        "logistic regression", "support vector machines", "svm", "k-means clustering",
        "hierarchical clustering", "pca", "dimensionality reduction", "feature engineering",
        "feature selection", "cross-validation", "hyperparameter tuning", "optuna",
        "grid search", "time series analysis", "arima", "prophet", "anomaly detection",
        "recommender systems", "collaborative filtering", "supervised learning",
        "unsupervised learning", "reinforcement learning", "q-learning"
    ],
    "MLOps, Deployment & Infrastructure": [
        "mlops", "mlflow", "kubeflow", "dvc", "data version control", "weights & biases",
        "wandb", "docker", "kubernetes", "ci/cd", "continuous integration", "github actions",
        "fastapi", "flask", "triton", "bentoml", "ray", "ray train", "ray serve",
        "model deployment", "model monitoring", "data drift", "concept drift",
        "model registry", "feature store", "feast", "onnx", "tensorrt", "torchscript",
        "pipeline orchestration", "airflow", "prefect", "dagster", "cuda", "gpu optimization"
    ],
    "Data Engineering & Big Data": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "apache spark", "pyspark",
        "databricks", "snowflake", "bigquery", "apache kafka", "data warehousing",
        "etl", "elt", "data pipelines", "pandas", "numpy", "polars", "dask",
        "data cleaning", "data preprocessing", "data modeling", "data governance"
    ],
    "Mathematics, Statistics & Analytics": [
        "statistics", "probability", "linear algebra", "calculus", "bayesian statistics",
        "hypothesis testing", "a/b testing", "statistical modeling", "p-values",
        "confidence intervals", "exploratory data analysis", "eda", "data visualization",
        "matplotlib", "seaborn", "plotly", "power bi", "tableau", "excel"
    ],
    "Cloud Computing & Platforms": [
        "aws", "aws sagemaker", "s3", "ec2", "lambda", "gcp", "google cloud platform",
        "vertex ai", "azure", "azure ml", "cloud architecture", "serverless", "terraform"
    ],
    "Software Engineering & Tools": [
        "python", "c++", "java", "scala", "r", "rust", "git", "github", "gitlab",
        "rest api", "graphql", "microservices", "object-oriented programming", "oop",
        "data structures", "algorithms", "unit testing", "pytest", "linux", "bash"
    ]
}

# Master list of all flattened skills for quick matching
ALL_TAXONOMY_SKILLS = []
for category, skills in CATEGORIZED_SKILLS.items():
    for skill in skills:
        if skill not in ALL_TAXONOMY_SKILLS:
            ALL_TAXONOMY_SKILLS.append(skill)

ROLE_BENCHMARKS = {
    "🤖 Generative AI / LLM Engineer": {
        "description": "Designs, fine-tunes, and deploys Large Language Models, RAG architectures, multi-agent systems, and autonomous AI pipelines into production.",
        "core_skills": [
            "python", "large language models", "llm", "rag", "langchain", "llamaindex",
            "prompt engineering", "fine-tuning", "lora", "vector databases", "huggingface",
            "transformers", "pytorch", "embeddings", "openai api", "fastapi"
        ],
        "secondary_skills": [
            "peft", "qlora", "vllm", "ollama", "agentic workflows", "crewai",
            "triton inference server", "pinecone", "chromadb", "docker", "semantic search",
            "rlhf", "evaluation metrics", "guardrails", "git"
        ],
        "category_weights": {
            "Generative AI & LLMs": 0.40,
            "Deep Learning & AI Architectures": 0.20,
            "MLOps, Deployment & Infrastructure": 0.20,
            "Software Engineering & Tools": 0.10,
            "Data Engineering & Big Data": 0.10
        },
        "target_keywords": [
            "LLM", "RAG", "Prompt Engineering", "Fine-tuning", "LangChain", "LlamaIndex",
            "PyTorch", "HuggingFace", "Vector Search", "Embeddings", "FastAPI", "Transformers"
        ]
    },
    "🧠 Machine Learning Engineer": {
        "description": "Builds, optimizes, and productionizes predictive and deep learning models at scale, bridging data science with robust software engineering.",
        "core_skills": [
            "python", "machine learning", "deep learning", "pytorch", "tensorflow",
            "scikit-learn", "feature engineering", "model training", "model evaluation",
            "docker", "fastapi", "git", "sql", "pandas", "numpy"
        ],
        "secondary_skills": [
            "xgboost", "lightgbm", "mlflow", "hyperparameter tuning", "optuna",
            "ci/cd", "kubernetes", "aws", "model deployment", "onnx", "rest api",
            "linux", "unit testing", "cross-validation"
        ],
        "category_weights": {
            "Classical Machine Learning & Modeling": 0.30,
            "Deep Learning & AI Architectures": 0.25,
            "MLOps, Deployment & Infrastructure": 0.20,
            "Software Engineering & Tools": 0.15,
            "Data Engineering & Big Data": 0.10
        },
        "target_keywords": [
            "Machine Learning", "PyTorch", "Scikit-Learn", "Model Training", "Deployment",
            "Feature Engineering", "FastAPI", "Docker", "MLflow", "Hyperparameter Tuning"
        ]
    },
    "⚙️ MLOps Engineer": {
        "description": "Automates the end-to-end ML lifecycle: continuous training, pipeline orchestration, model registry, monitoring, scaling, and GPU infrastructure.",
        "core_skills": [
            "python", "mlops", "docker", "kubernetes", "mlflow", "kubeflow",
            "ci/cd", "fastapi", "git", "aws", "linux", "model deployment",
            "model monitoring", "airflow"
        ],
        "secondary_skills": [
            "dvc", "weights & biases", "triton", "bentoml", "ray", "terraform",
            "data drift", "prometheus", "grafana", "s3", "github actions", "cuda",
            "pytorch", "bash"
        ],
        "category_weights": {
            "MLOps, Deployment & Infrastructure": 0.45,
            "Cloud Computing & Platforms": 0.20,
            "Software Engineering & Tools": 0.20,
            "Classical Machine Learning & Modeling": 0.10,
            "Data Engineering & Big Data": 0.05
        },
        "target_keywords": [
            "MLOps", "Kubernetes", "Docker", "MLflow", "CI/CD", "Model Monitoring",
            "Airflow", "FastAPI", "Pipeline", "Infrastructure", "Kubeflow"
        ]
    },
    "🔬 Data Scientist": {
        "description": "Derives strategic insights from complex data through statistical rigor, predictive modeling, machine learning, and experimental design.",
        "core_skills": [
            "python", "sql", "statistics", "machine learning", "pandas", "numpy",
            "scikit-learn", "data visualization", "exploratory data analysis",
            "hypothesis testing", "a/b testing", "data cleaning"
        ],
        "secondary_skills": [
            "matplotlib", "seaborn", "plotly", "xgboost", "random forest", "deep learning",
            "tableau", "power bi", "r", "predictive modeling", "feature engineering",
            "time series analysis", "bigquery"
        ],
        "category_weights": {
            "Mathematics, Statistics & Analytics": 0.35,
            "Classical Machine Learning & Modeling": 0.30,
            "Data Engineering & Big Data": 0.20,
            "Software Engineering & Tools": 0.15
        },
        "target_keywords": [
            "Data Science", "Statistics", "A/B Testing", "SQL", "Pandas", "Scikit-Learn",
            "Predictive Modeling", "Hypothesis Testing", "EDA", "Visualization"
        ]
    },
    "👁️ Computer Vision Engineer": {
        "description": "Develops state-of-the-art vision models for object detection, segmentation, image processing, OCR, and real-time edge/cloud inference.",
        "core_skills": [
            "python", "computer vision", "opencv", "pytorch", "deep learning",
            "cnn", "convolutional neural networks", "yolo", "object detection",
            "image segmentation", "tensorflow"
        ],
        "secondary_skills": [
            "vision transformers", "tensorrt", "cuda", "onnx", "data preprocessing",
            "c++", "docker", "edge ai", "model optimization", "transfer learning"
        ],
        "category_weights": {
            "Deep Learning & AI Architectures": 0.50,
            "Software Engineering & Tools": 0.20,
            "MLOps, Deployment & Infrastructure": 0.15,
            "Mathematics, Statistics & Analytics": 0.15
        },
        "target_keywords": [
            "Computer Vision", "OpenCV", "PyTorch", "YOLO", "Object Detection",
            "Image Segmentation", "CNN", "TensorRT", "Deep Learning"
        ]
    },
    "🗣️ NLP / Speech Engineer": {
        "description": "Specializes in text and audio processing, tokenization, transformers, conversational AI, semantic parsing, and acoustic modeling.",
        "core_skills": [
            "python", "nlp", "natural language processing", "pytorch", "transformers",
            "huggingface", "bert", "spacy", "deep learning", "tokenization", "text classification"
        ],
        "secondary_skills": [
            "named entity recognition", "sentiment analysis", "nltk", "whisper",
            "speech recognition", "large language models", "fine-tuning", "regex",
            "fastapi", "embeddings"
        ],
        "category_weights": {
            "Deep Learning & AI Architectures": 0.40,
            "Generative AI & LLMs": 0.25,
            "Classical Machine Learning & Modeling": 0.15,
            "Software Engineering & Tools": 0.20
        },
        "target_keywords": [
            "NLP", "Transformers", "HuggingFace", "BERT", "SpaCy", "PyTorch",
            "Tokenization", "NER", "Text Processing", "Large Language Models"
        ]
    },
    "📊 Data Analyst / BI Specialist": {
        "description": "Transforms business requirements into actionable dashboards, SQL analytics, data pipelines, metric definitions, and executive presentations.",
        "core_skills": [
            "sql", "excel", "power bi", "tableau", "data visualization", "pandas",
            "data cleaning", "statistics", "exploratory data analysis", "reporting"
        ],
        "secondary_skills": [
            "python", "numpy", "postgresql", "bigquery", "etl", "data modeling",
            "kpi tracking", "dashboards", "business intelligence", "presentation"
        ],
        "category_weights": {
            "Mathematics, Statistics & Analytics": 0.40,
            "Data Engineering & Big Data": 0.35,
            "Software Engineering & Tools": 0.25
        },
        "target_keywords": [
            "SQL", "Power BI", "Tableau", "Data Visualization", "Pandas",
            "Business Intelligence", "Reporting", "Dashboards", "Excel"
        ]
    },
    "🌐 Full-Stack AI Engineer": {
        "description": "Builds end-to-end user-facing AI applications, integrating modern frontend frameworks with robust backend APIs and LLM/ML services.",
        "core_skills": [
            "python", "fastapi", "javascript", "react", "large language models",
            "rest api", "docker", "sql", "git", "langchain", "prompt engineering"
        ],
        "secondary_skills": [
            "next.js", "typescript", "vector databases", "pinecone", "rag",
            "mongodb", "postgresql", "tailwind css", "ci/cd", "aws", "openai api"
        ],
        "category_weights": {
            "Software Engineering & Tools": 0.35,
            "Generative AI & LLMs": 0.25,
            "MLOps, Deployment & Infrastructure": 0.20,
            "Data Engineering & Big Data": 0.20
        },
        "target_keywords": [
            "Full Stack", "FastAPI", "React", "LLM", "REST API", "LangChain",
            "Docker", "TypeScript", "Python", "RAG"
        ]
    }
}

SKILL_LEARNING_RESOURCES = {
    "large language models": {"title": "DeepLearning.AI: Generative AI with LLMs", "link": "https://www.deeplearning.ai/courses/generative-ai-with-llms/"},
    "rag": {"title": "LangChain & LlamaIndex RAG Masterclass", "link": "https://docs.llamaindex.ai/"},
    "langchain": {"title": "LangChain Official Tutorials", "link": "https://python.langchain.com/docs/tutorials/"},
    "llamaindex": {"title": "LlamaIndex Documentation & Starter Guide", "link": "https://docs.llamaindex.ai/"},
    "prompt engineering": {"title": "OpenAI Prompt Engineering Guide", "link": "https://platform.openai.com/docs/guides/prompt-engineering"},
    "fine-tuning": {"title": "Hugging Face PEFT & LoRA Guide", "link": "https://huggingface.co/docs/peft"},
    "vector databases": {"title": "Pinecone / Chroma Vector Search Guide", "link": "https://www.pinecone.io/learn/"},
    "pytorch": {"title": "PyTorch Official Deep Learning Tutorials", "link": "https://pytorch.org/tutorials/"},
    "deep learning": {"title": "Deep Learning Specialization by Andrew Ng", "link": "https://www.coursera.org/specializations/deep-learning"},
    "machine learning": {"title": "Machine Learning Specialization by Andrew Ng", "link": "https://www.coursera.org/specializations/machine-learning-introduction"},
    "mlops": {"title": "Made With ML: Production MLOps Course", "link": "https://madewithml.com/"},
    "mlflow": {"title": "MLflow Documentation & Quickstart", "link": "https://mlflow.org/docs/latest/index.html"},
    "docker": {"title": "Docker for Data Science & ML Engineers", "link": "https://docs.docker.com/get-started/"},
    "kubernetes": {"title": "Kubernetes Official Tutorials", "link": "https://kubernetes.io/docs/tutorials/"},
    "fastapi": {"title": "FastAPI Web Framework Documentation", "link": "https://fastapi.tiangolo.com/"},
    "computer vision": {"title": "CS231n: Deep Learning for Computer Vision (Stanford)", "link": "https://cs231n.stanford.edu/"},
    "opencv": {"title": "OpenCV Python Tutorials", "link": "https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html"},
    "yolo": {"title": "Ultralytics YOLO Documentation", "link": "https://docs.ultralytics.com/"},
    "nlp": {"title": "Hugging Face NLP Course", "link": "https://huggingface.co/learn/nlp-course"},
    "transformers": {"title": "Hugging Face Transformers Documentation", "link": "https://huggingface.co/docs/transformers"},
    "sql": {"title": "Mode Analytics SQL Tutorial for Data Analysis", "link": "https://mode.com/sql-tutorial/"},
    "pandas": {"title": "Pandas Official Documentation & 10-Minute Guide", "link": "https://pandas.pydata.org/docs/user_guide/10min.html"},
    "statistics": {"title": "Khan Academy Statistics & Probability", "link": "https://www.khanacademy.org/math/statistics-probability"},
    "aws sagemaker": {"title": "AWS Machine Learning Specialty Training", "link": "https://aws.amazon.com/sagemaker/"},
    "ci/cd": {"title": "GitHub Actions Documentation for CI/CD", "link": "https://docs.github.com/en/actions"}
}
