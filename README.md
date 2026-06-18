# Airline Complaint Classification System

## Overview

The Airline Complaint Classification System is a Natural Language Processing (NLP) and Deep Learning project that automatically categorizes airline customer complaints into predefined complaint categories.

The system leverages text preprocessing, data augmentation, GloVe word embeddings, and a Bidirectional GRU neural network to classify customer complaints efficiently and accurately.

---

## Problem Statement

Airlines receive a large volume of customer complaints related to flight delays, baggage handling, refunds, customer service, seat reservations, and other operational issues. Manually analyzing and categorizing these complaints is time-consuming and error-prone.

This project automates the complaint categorization process, enabling faster issue routing and improved customer support operations.

---

## Features

- Text Cleaning and Preprocessing
- Stopword Removal and Lemmatization
- Data Augmentation using NLPAug
- Class Imbalance Handling using Class Weights and Focal Loss
- Pre-trained GloVe Word Embeddings
- Bidirectional GRU Deep Learning Architecture
- Multi-Class Complaint Classification
- Model Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-Learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLPAug
- Joblib

---

## Project Workflow

```text
Raw Complaint Text
        ↓
Text Preprocessing
        ↓
Data Augmentation
        ↓
Tokenization
        ↓
GloVe Embeddings
        ↓
Bidirectional GRU Model
        ↓
Complaint Category Prediction
```

---

## Model Architecture

- Embedding Layer (GloVe 100-Dimensional Embeddings)
- Spatial Dropout Layer
- Bidirectional GRU Layer
- Global Max Pooling Layer
- Dense Layer (128 Units)
- Dense Layer (64 Units)
- Softmax Output Layer

---

## Training Techniques

- Adam Optimizer
- Sparse Categorical Focal Loss
- Class Weights
- Early Stopping
- Reduce Learning Rate on Plateau

---

## Results

| Model | Accuracy |
|---------|---------|
| Feed Forward Network (FFN) | 72% |
| Bidirectional LSTM | 78% |
| Bidirectional GRU | **88%** |

The Bidirectional GRU model achieved the best performance and was selected as the final model.

---

## Sample Predictions

### Input

```text
My flight was delayed for 5 hours
```

### Output

```text
Flight Delay
```

### Input

```text
My baggage was lost at the airport
```

### Output

```text
Baggage Issue
```

### Input

```text
Customer support ignored my refund request
```

### Output

```text
Refund Related Complaint
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Mathin21/airline-complaint-classification.git
```

Navigate to the project directory:

```bash
cd airline-complaint-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Model Training

```bash
python classification.py
```

---

## Generated Files

After training, the following files are generated:

```text
airline_complaint_classifier.keras
tokenizer.pkl
label_encoder.pkl
airline_complaint_classifier.weights.h5
```

---

## Future Improvements

- BERT-based Classification Models
- RoBERTa and Transformer Architectures
- Explainable AI Techniques
- Real-Time Dashboard for Complaint Analytics
- Cloud Deployment

---

## Author

**Shaik Mathin**

- LinkedIn: https://linkedin.com/in/shaik-mathin-a60358203
- GitHub: https://github.com/Mathin21

---

## License

This project is intended for educational and research purposes.
