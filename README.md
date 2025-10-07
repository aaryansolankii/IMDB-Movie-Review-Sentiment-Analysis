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
- Accuracy: 0.9016
- F1-score: 0.90

---

## 🤗 Transformer Model (DistilBERT)
- **Fine-tuning:** Hugging Face `Trainer` API  
- **Training parameters:**
  - Learning rate: 2e-5  
  - Batch size: train=8, eval=16  
  - Epochs: 3  
  - Weight decay: 0.01  

- **Test Set Results:**
- Accuracy: 0.9174
- Precision: 0.9175
- Recall: 0.9174
- F1-score: 0.9174

- **Confusion Matrix:**
-[[4558 442]
-[ 384 4616]]


---

## 💾 Model Optimization
- Original size: ~250 MB  
- Quantized to 8-bit → ~87 MB  
- Faster loading and lower memory footprint  
- Suitable for **Streamlit or cloud deployment**

---

## 🖥️ Deployment
- **Streamlit-ready pipeline** using Hugging Face `pipeline` API  
- Can run on GPU or CPU  
