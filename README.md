# SMS-Spam-Detection

This project is an AI-powered spam detection system that classifies emails and SMS as Spam or Not Spam (Ham) using machine learning and natural language processing. Messages are processed with TF-IDF, analyzed by ML models, and deployed via a modern Streamlit web app for real-time predictions.

[![Streamlit App](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-2B82F6?style=for-the-badge)](https://sms-spam-detection-001.streamlit.app/)

# **📓 Project Report: Email Spam Detection with Machine Learning**

# **1. Introduction**

Spam emails (junk emails) are unwanted messages that are often sent in bulk. They can contain advertisements, scams, phishing attempts, or malware links, making spam detection a critical problem in cybersecurity.

This project aims to build a Machine Learning-based Spam Detection System that automatically classifies emails or SMS messages into two categories:

Ham (Not Spam) → Normal/legitimate messages.

Spam → Unwanted, malicious, or irrelevant messages.

# **2. Problem Definition**

We frame this as a binary text classification problem:

Input: A raw text message (email or SMS).

Output: A predicted label → Spam (1) or Ham (0).

Challenges include:

Spam messages are often disguised to look legitimate.

Messages are short and use informal language (abbreviations, misspellings).

Spammers constantly evolve tactics, requiring adaptable models.

# **3. Dataset**

We used the UCI SMS Spam Collection Dataset (available on Kaggle).

Total ~5,500 messages labeled as ham or spam.

Each record contains:

Label: "ham" (not spam) or "spam".

Message: The text of the SMS/email.

This dataset is widely used for spam detection research.

# **4. Workflow & Methodology**

**Step 1: Data Preprocessing**

Label Encoding → Convert "ham" → 0, "spam" → 1.

Text Cleaning → Removing stopwords, punctuation, numbers, and lowercasing text.

Train-Test Split → Dividing dataset into training (80%) and testing (20%).

**Step 2: Feature Extraction (NLP)**

Since machine learning models cannot understand raw text, we convert text into numerical features.

TF-IDF (Term Frequency–Inverse Document Frequency) is used.

It assigns higher weight to words that are important to a message but less frequent across all messages.

Example: Words like "free", "win", "urgent" appear more often in spam, hence weighted strongly.

**Step 3: Model Training**

We train multiple classification models:

*Naïve Bayes*

Probabilistic model based on Bayes’ theorem.

Works well for text data.

Fast and lightweight → good baseline.

*Logistic Regression*

Linear model for classification.

Handles high-dimensional sparse data effectively.

*Support Vector Machine (SVM)*

Finds a decision boundary (hyperplane) between spam and ham.

Good for text classification with high accuracy.

# **Step 4: Model Evaluation**

We use standard metrics to evaluate performance:

Accuracy → Overall correctness.

Precision → How many predicted spams are actually spam.

Recall (Sensitivity) → How many actual spams were correctly detected.

F1-Score → Harmonic mean of precision and recall.

Confusion Matrix → Visualization of True/False positives and negatives.

# **Step 5: Deployment with Streamlit**

We deploy the best-performing model as a web app using Streamlit.

User inputs a message in a text box.

Model predicts whether it is Spam or Not Spam.

Confidence score is displayed.

# **5. Results**

All models perform well, with SVM and Logistic Regression achieving >95% accuracy.

Naïve Bayes is slightly less accurate but much faster to train.

The confusion matrix shows very few misclassifications.

Typical outcomes:

Ham Example: "Let’s meet at 5 pm for the project." → Predicted: Not Spam ✅

Spam Example: "Congratulations! You’ve won a free vacation. Click here!" → Predicted: Spam 🚨

# **6. Applications**

Email providers (Gmail, Yahoo) → Filtering spam from inbox.

Telecom companies → Detecting fraudulent SMS.

Cybersecurity → Prevent phishing attacks.

Enterprise systems → Reducing unwanted bulk messages.

# **7. Future Improvements**

Use deep learning (LSTM, GRU, BERT) for contextual understanding.

Include metadata (sender email, number of links, subject length, etc.).

Build an adaptive learning system that updates with new spam patterns.

Extend to multi-language spam detection.

# **8. Conclusion**

This project successfully demonstrates how machine learning and natural language processing (NLP) can be applied to classify emails/SMS into spam and ham.
The system achieves high accuracy and can be deployed as a real-time spam detection tool using Streamlit.

It’s lightweight, interpretable, and can serve as a foundation for more advanced AI-driven spam filtering systems.
