# 🔍 AuthentiText: Detecting AI-Generated Text Using ML & Feature Engineering
With the rise of large language models (LLMs), distinguishing AI-generated text from human writing has become critical—especially in academia, journalism, and content authenticity. AuthentiText is a machine learning project designed to address this challenge by building an advanced classification pipeline for AI vs. human text detection.

We experimented with multiple approaches:

1. ✅ Baseline models like logistic regression, which struggled with the complexity of the data.

2. 🧠 Transformer models (BERT) and Random Forest ensembles, which showed promise but lacked robustness.

3. 🧪 High-impact feature engineering, where we achieved ~97% accuracy using a Random Forest model enriched with linguistic and semantic features like sentiment polarity, vocabulary richness, named entities, and readability metrics.

To make the system accessible, we also built a Streamlit web app that highlights impactful words in the prediction and provides insights into linguistic patterns—helping users understand why a piece of text is classified a certain way.

🔗 GitHub: AuthentiText Repo

🖥️ Live Demo: Try the App
