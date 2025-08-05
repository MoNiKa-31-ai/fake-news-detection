#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm


# In[4]:


# Load Fake and True news
fake = pd.read_csv(r"C:\Users\HP\Downloads\archive\News _dataset\Fake.csv")
true = pd.read_csv(r"C:\Users\HP\Downloads\archive\News _dataset\True.csv")
fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)
data["content"] = data["title"] + " " + data["text"]

# Limit to 500 for speed
N = 500
data = data.iloc[:N]
texts = data["content"].tolist()
labels = data["label"].tolist()


# In[5]:


# TF-IDF Features
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_tfidf = tfidf.fit_transform(texts)

# Train-test split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Logistic Regression
lr_tfidf = LogisticRegression()
lr_tfidf.fit(X_train_tfidf, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predict on test set
y_pred = lr_tfidf.predict(X_test_tfidf)

# Print metrics
print("📊 Performance Metrics for TF-IDF + Logistic Regression:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("\nFull Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))


# In[7]:


# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Extract CLS Embeddings
def get_bert_embeddings(texts):
    embeddings = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        cls_embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding.squeeze().cpu().numpy())
    return np.array(embeddings)

X_bert = get_bert_embeddings(texts)

# Train-test split
X_train_bert, X_test_bert, _, _ = train_test_split(X_bert, labels, test_size=0.2, random_state=42)

# Random Forest on BERT
rf_bert = RandomForestClassifier()
rf_bert.fit(X_train_bert, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predict on test set
y_pred = rf_bert.predict(X_test_bert)

# Print metrics
print("📊 Performance Metrics for TF-IDF + Logistic Regression:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("\nFull Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))



# In[8]:


# Predict probabilities
lr_probs = lr_tfidf.predict_proba(X_test_tfidf)[:,1]
rf_probs = rf_bert.predict_proba(X_test_bert)[:,1]

# Average probabilities (Soft voting)
final_probs = (lr_probs + rf_probs) / 2
final_preds = (final_probs >= 0.5).astype(int)

# Evaluate
print("✅ Ensemble Voting Accuracy:", accuracy_score(y_test, final_preds))
print("Precision:", round(precision_score(y_test,final_preds ), 4))
print("Recall:", round(recall_score(y_test, final_preds), 4))
print("F1 Score:", round(f1_score(y_test, final_preds), 4))
print("\nFull Classification Report:\n")
print(classification_report(y_test, final_preds, target_names=["Fake", "Real"]))


# In[9]:


from sklearn.linear_model import LogisticRegressionCV

# Base model predictions as features
X_meta_train = np.vstack([
    lr_tfidf.predict_proba(X_train_tfidf)[:,1],
    rf_bert.predict_proba(X_train_bert)[:,1]
]).T

X_meta_test = np.vstack([
    lr_probs,
    rf_probs
]).T

# Meta model
meta_model = LogisticRegressionCV(cv=5)
meta_model.fit(X_meta_train, y_train)

# Final stacked prediction
meta_preds = meta_model.predict(X_meta_test)

# Evaluate
print("✅ Stacking Ensemble Accuracy:", accuracy_score(y_test, meta_preds))
print("Precision:", round(precision_score(y_test,meta_preds ), 4))
print("Recall:", round(recall_score(y_test, meta_preds), 4))
print("F1 Score:", round(f1_score(y_test, meta_preds), 4))
print("\nFull Classification Report:\n")
print(classification_report(y_test, meta_preds, target_names=["Fake", "Real"]))


# In[8]:


get_ipython().system('pip install matplotlib scikit-learn')



# In[10]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# In[11]:


# Predict
y_pred_lr = lr_tfidf.predict(X_test_tfidf)

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=["Fake", "Real"])
disp_lr.plot(cmap="Blues")
plt.title("Confusion Matrix: TF-IDF + Logistic Regression")
plt.show()


# In[11]:


# Predict
y_pred_rf = rf_bert.predict(X_test_bert)

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Fake", "Real"])
disp_rf.plot(cmap="Greens")
plt.title("Confusion Matrix: BERT + Random Forest")
plt.show()


# In[12]:


# Voting predictions were already calculated (final_preds)
cm_voting = confusion_matrix(y_test, final_preds)
disp_voting = ConfusionMatrixDisplay(confusion_matrix=cm_voting, display_labels=["Fake", "Real"])
disp_voting.plot(cmap="Oranges")
plt.title("Confusion Matrix: Voting Ensemble")
plt.show()


# In[13]:


# Stacking predictions (meta_preds) were already calculated
cm_stack = confusion_matrix(y_test, meta_preds)
disp_stack = ConfusionMatrixDisplay(confusion_matrix=cm_stack, display_labels=["Fake", "Real"])
disp_stack.plot(cmap="Purples")
plt.title("Confusion Matrix: Stacking Ensemble")
plt.show()


# In[14]:


# Predict probabilities
lr_probs = lr_tfidf.predict_proba(X_test_tfidf)[:,1]
rf_probs = rf_bert.predict_proba(X_test_bert)[:,1]
# Voting already averaged
# Stacking already predicted with meta_model

# ROC Curve for TF-IDF + LR
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# ROC Curve for BERT + RF
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# ROC Curve for Voting Ensemble
fpr_vote, tpr_vote, _ = roc_curve(y_test, final_probs)
roc_auc_vote = auc(fpr_vote, tpr_vote)

# ROC Curve for Stacking Ensemble
fpr_stack, tpr_stack, _ = roc_curve(y_test, meta_model.predict_proba(X_meta_test)[:,1])
roc_auc_stack = auc(fpr_stack, tpr_stack)

# Plot
plt.figure(figsize=(10,8))
plt.plot(fpr_lr, tpr_lr, label=f'TF-IDF + LR (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'BERT + RF (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_vote, tpr_vote, label=f'Voting Ensemble (AUC = {roc_auc_vote:.2f})')
plt.plot(fpr_stack, tpr_stack, label=f'Stacking Ensemble (AUC = {roc_auc_stack:.2f})')

# Random baseline
plt.plot([0,1], [0,1], 'k--')

plt.title('ROC Curves Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[ ]:




