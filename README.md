🧠 **Hybrid Fine-Grained NER System for Low-Resource Telugu Text**
**Semi-Supervised & Explainable Learning Approach**

📌 **Project Overview**

This project presents a Hybrid Fine-Grained Named Entity Recognition (NER) System designed specifically for low-resource Telugu text.

The system leverages:

Transformer-based models

Semi-Supervised Learning

Explainable AI (LIME)

It identifies fine-grained entities such as:

👤 PER (Person)

📍 LOC (Location)

🏢 ORG (Organization)

The model is trained using Telugu Wikipedia data and the naamapadam dataset, and deployed with a Streamlit-based interactive UI.

🎯 **Problem Statement**

Low-resource languages like Telugu face:

Limited annotated datasets

Poor generalization in traditional NER models

Lack of explainability in predictions

This project addresses these challenges by combining semi-supervised learning with explainable AI techniques.

🚀 **Key Features**

📚 Fine-grained entity recognition for Telugu

🔄 Semi-Supervised learning to leverage unlabeled data

🔍 Transformer-based model (HuggingFace)

🧠 Explainable AI using LIME

🎨 Interactive Streamlit frontend

💾 Outputs labeled entities (PER, LOC, ORG) instead of numeric IDs

🛠️ **Tech Stack**
Category	Technology
Language	Python
Explainability	LIME
UI	Streamlit
Dataset	Telugu Wikipedia + naamapadam
Notebook	Jupyter

🧠 **System Architecture**
Input Telugu Text
        ↓
Tokenizer (AutoTokenizer)
        ↓
Transformer Model
        ↓
Fine-Grained Entity Prediction
        ↓
LIME Explainability Layer
        ↓
Streamlit UI Output

📊 **Model Details**

Tokenization: AutoTokenizer

Architecture: Transformer-based token classification

Learning Type: Semi-Supervised

**Evaluation Metrics:**

Precision

Recall

F1-Score

🔍 **Explainability (LIME Integration)**

LIME is integrated to:

Highlight important tokens influencing predictions

Improve transparency

Increase trust in model decisions

This makes the system suitable for research and real-world NLP applications.

📈 **Results**

Improved performance on low-resource Telugu dataset

Accurate entity labeling (PER, LOC, ORG)

Transparent prediction explanations using LIME

🌟 **Project Highlights**

Research-oriented NLP project

Handles low-resource language challenges

Combines semi-supervised + explainable learning

End-to-end implementation (Training → Testing → Deployment)

🔮 **Future Enhancements**

Multi-language extension

Active learning integration

Deployment as REST API

Real-time Telugu news NER system

Integration with knowledge graphs

👩‍💻 **Developed By**

Yasaswi Challa
