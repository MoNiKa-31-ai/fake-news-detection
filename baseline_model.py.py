#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from scipy.sparse import hstack
import numpy as np


# In[3]:


# Load Fake and True news
fake = pd.read_csv(r"C:\Users\HP\Downloads\archive\News _dataset\Fake.csv")
true = pd.read_csv(r"C:\Users\HP\Downloads\archive\News _dataset\True.csv")

# Add a label column
fake['label'] = 0  # Fake = 0
true['label'] = 1  # Real = 1

# Combine them
data = pd.concat([fake, true], axis=0)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# View few samples
print(data.head())


# In[4]:


# Combine title and text
data['content'] = data['title'] + " " + data['text']

# Define features and labels
X_texts = data['content']
y = data['label']


# In[5]:



# Create TF-IDF features (unigrams + bigrams)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X_texts)

print("TF-IDF shape:", X_tfidf.shape)


# In[6]:


# Function to get sentiment
def get_sentiment_features(text_list):
    polarity = []
    subjectivity = []
    for text in text_list:
        blob = TextBlob(str(text))
        polarity.append(blob.sentiment.polarity)
        subjectivity.append(blob.sentiment.subjectivity)
    return np.array(polarity).reshape(-1,1), np.array(subjectivity).reshape(-1,1)

# Apply function
polarity, subjectivity = get_sentiment_features(X_texts)

print("Polarity shape:", polarity.shape)


# In[7]:


np.random.seed(42)  # For reproducibility

# Simulated metadata: domain trust (0 to 100) and random follower counts
domain_authority = np.random.randint(10, 100, size=(len(X_texts), 1))
follower_count = np.random.randint(100, 100000, size=(len(X_texts), 1))

# Stack simulated metadata
metadata_features = np.hstack([domain_authority, follower_count])

print("Metadata features shape:", metadata_features.shape)


# In[8]:


# Combine TF-IDF + Sentiment + Metadata
X_combined = hstack([X_tfidf, polarity, subjectivity, metadata_features])

print("Final Combined Feature Matrix Shape:", X_combined.shape)


# In[9]:


# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)


# In[10]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict and get confusion matrix
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:



get_ipython().system('pip install transformers torch tqdm')


# In[14]:


from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm



# In[15]:


import pandas as pd

# Load Fake and True news
fake = pd.read_csv(r"C:\Users\HP\Downloads\archive\News _dataset\Fake.csv")
true = pd.read_csv(r"C:\Users\HP\Downloads\archive\News _dataset\True.csv")

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine and shuffle
data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Use first N samples for demo (BERT is heavy)
N = 500  # adjust this depending on memory
data = data.iloc[:N]

# Combine title and text
data['content'] = data['title'] + " " + data['text']
texts = list(data['content'])
labels = list(data['label'])


# In[16]:


# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set model to evaluation mode
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to extract BERT embeddings (CLS token)
def get_bert_embeddings(texts):
    embeddings = []
    
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embedding.squeeze().cpu().numpy())
    
    return np.array(embeddings)

# Get embeddings
bert_features = get_bert_embeddings(texts)

print("BERT Embedding Shape:", bert_features.shape)  # Should be (N, 768)



# In[17]:


from textblob import TextBlob
import numpy as np

def get_sentiment_features(text_list):
    polarity = []
    subjectivity = []
    for text in text_list:
        blob = TextBlob(str(text))
        polarity.append(blob.sentiment.polarity)
        subjectivity.append(blob.sentiment.subjectivity)
    return np.array(polarity).reshape(-1,1), np.array(subjectivity).reshape(-1,1)

polarity, subjectivity = get_sentiment_features(texts)


# In[18]:


np.random.seed(42)
domain_authority = np.random.randint(10, 100, size=(len(texts), 1))
follower_count = np.random.randint(100, 100000, size=(len(texts), 1))
metadata_features = np.hstack([domain_authority, follower_count])


# In[19]:


from sklearn.model_selection import train_test_split

# Final Feature Matrix
X_all = np.hstack([bert_features, polarity, subjectivity, metadata_features])
y_all = np.array(labels)

print("Final Feature Matrix Shape:", X_all.shape)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy with BERT + Sentiment + Metadata:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict and get confusion matrix
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:




