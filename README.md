# 📰 Fake News Detection using TF-IDF, TextBlob & BERT Embeddings

This project explores a hybrid ensemble-based fake news classification system. We combine shallow (TF-IDF, sentiment) and deep (BERT embeddings) features to detect fake vs real news with high accuracy.

---

## Techniques Used

- **TF-IDF + Logistic Regression**
- **BERT embeddings + Random Forest**
- **Ensemble Voting and Stacking**
- **Sentiment features from TextBlob**
- **Simulated metadata (follower count, domain trust)**

---

##  Files

| File | Description |
|------|-------------|
| `baseline_model.py` | TF-IDF, TextBlob sentiment, metadata + Random Forest |
| `ensemble_model.py` | TF-IDF + BERT + Ensemble voting + stacking |
| `research_paper_draft.docx` | Course-based unpublished research draft |

---

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Curve

---

## Dataset

Used the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), including:

- `Fake.csv` → label: 0
- `True.csv` → label: 1

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 96% | 0.95 |
| BERT + Random Forest | 95% | 0.94 |
| **Voting Ensemble** | **98%** | 0.97 |
| **Stacking Ensemble** | **95%** | 0.95 |

---

##  Research Summary

This project was part of a course-based research paper exploring hybrid models for fake news detection. Techniques from both traditional ML and deep learning were compared and combined for better generalization and robustness.

---

##  License

This work is academic in nature and not intended for commercial use.
