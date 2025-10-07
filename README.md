# IMDB Movie Review Sentiment Analysis 🎬💬

A complete **sentiment analysis system** for IMDB movie reviews using both **traditional ML** and **Transformer-based models (DistilBERT)**.  
Classifies reviews as **Positive** or **Negative** with high accuracy and is ready for **Streamlit deployment**.

---

## 🚀 Project Overview
This project demonstrates:
- Data preprocessing & cleaning  
- Baseline model with **TF-IDF + Logistic Regression**  
- Transformer fine-tuning (**DistilBERT**)  
- Model quantization (8-bit) to reduce size for deployment  
- Streamlit-ready inference pipeline

---

## 📊 Dataset
- **Source:** [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Size:** 50,000 reviews (balanced: 25k positive, 25k negative)  
- **Columns:**
  - `review` → text of the review
  - `sentiment` → positive/negative

---

## 🧹 Data Preprocessing
1. Rename columns: `review` → `text`, `sentiment` → `label`  
2. Drop missing values  
3. Trim extra spaces  
4. Map labels to integers: `negative=0`, `positive=1`  
5. Split dataset:  
   - Train: 70%  
   - Validation: 10%  
   - Test: 20%  

---

## 🧮 Baseline Model
- **Pipeline:** TF-IDF + Logistic Regression  
- **TF-IDF:** `max_features=50000`, `ngram_range=(1,2)`  
- **Logistic Regression:** `saga solver`, `max_iter=1000`  
- **Validation Results:**
