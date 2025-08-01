
## 📚 Movie Genre Classification Using Deep Learning

### 🔍 Overview

This project aims to classify movie genres based on text data such as plot summaries or metadata using Natural Language Processing (NLP) and Deep Learning techniques. Achieved **82% accuracy** using a Bidirectional LSTM model with proper preprocessing, tokenization, and class balancing.

---

### 🧠 Problem Statement

Given movie descriptions, the task is to predict the most likely **genre**. This is a **multi-class text classification** problem.

---

### 📊 Dataset

* Source: `movie_data.csv`
* Contains:

  * Column 0: Movie description or plot
  * Column 1: Genre label (e.g., Action, Comedy)

---

### 🛠️ Technologies Used

* Python
* TensorFlow & Keras
* NLTK
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook

---

### 🔄 Workflow

1. **Data Preprocessing**

   * Cleaning text (lowercasing, punctuation removal)
   * Stopword removal
   * Lemmatization using WordNet

2. **Tokenization & Padding**

   * Keras Tokenizer with vocab size = 10,000
   * Padding sequences to maximum length

3. **Label Encoding**

   * LabelEncoder + One-hot encoding

4. **Model Architecture**

   * Embedding Layer
   * Bidirectional LSTM
   * GlobalMaxPooling
   * Dense + Dropout
   * Output layer with Softmax

5. **Training**

   * Optimizer: Adam
   * Epochs: 15
   * Validation split: 20%
   * EarlyStopping and class\_weight used

---

### ✅ Results

* **Validation Accuracy**: \~82%
* **Confusion Matrix**, **Classification Report**, and **Scatter Plot** used for evaluation.

---

### 📈 Visualizations

* Confusion Matrix (Seaborn heatmap)
* Actual vs Predicted Line Plot (for first 60 samples)
* Accuracy / Loss curves per epoch



### 📌 Future Improvements

* Use pre-trained embeddings (e.g., GloVe)
* Try Transformer-based models (e.g., BERT)
* Fine-tune for multi-label classification

