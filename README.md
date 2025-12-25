# Sentiment Analysis on Twitter (Sentiment140)

🚀 Binary sentiment classification on Twitter data using **Deep Learning** and **NLP**, with a comparative study between a **BiLSTM with GloVe embeddings** and a **Transformer-based model (DistilBERT)**.

---

## 📌 Project Overview

This project implements and compares two deep learning approaches for **sentiment analysis (positive vs negative)** on the **Sentiment140 Twitter dataset** (1.6M tweets):

- **BiLSTM + GloVe embeddings** (classical NLP pipeline)
- **DistilBERT fine-tuning** using PyTorch and GPU acceleration

The notebook highlights differences in preprocessing, training efficiency, and model performance between recurrent neural networks and transformer-based architectures.

---

## 🎯 Objectives

- Perform large-scale sentiment analysis on Twitter data  
- Apply and compare classical NLP vs Transformer-based approaches  
- Build efficient training pipelines for deep learning models  
- Evaluate performance trade-offs between BiLSTM and DistilBERT  

---

## 🧠 Models Implemented

### 1️⃣ BiLSTM with GloVe (TensorFlow)
- Text preprocessing (cleaning, stopwords removal, lemmatization)
- Tokenization & padding
- Pre-trained **GloVe 100d embeddings**
- **Bidirectional LSTM** architecture
- Binary classification (positive / negative)

### 2️⃣ DistilBERT (PyTorch + Transformers)
- Minimal text preprocessing
- Hugging Face `transformers` library
- Fine-tuning on Sentiment140
- GPU-accelerated training
- State-of-the-art Transformer-based NLP model

---

## 🛠️ Technologies & Tools

- **Python**
- **Pandas, NumPy**
- **NLTK**
- **TensorFlow / Keras**
- **PyTorch**
- **Hugging Face Transformers**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**
- **GPU (CUDA)**

---

## 📂 Dataset

- **Sentiment140 Twitter Dataset**
- 1.6 million labeled tweets
- Labels:
  - `0` → Negative
  - `1` → Positive
- Source: Kaggle

---

## ⚙️ Workflow

1. Dataset loading & exploration  
2. Text preprocessing (classical NLP)  
3. Train / validation / test split  
4. Tokenization & padding  
5. GloVe embeddings integration  
6. BiLSTM model training & evaluation  
7. DistilBERT fine-tuning with PyTorch  
8. Performance comparison & analysis  

---

## 📊 Evaluation

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

The comparison highlights:
- Faster convergence and stronger contextual understanding with Transformers
- Higher preprocessing dependency for classical BiLSTM pipelines

---

