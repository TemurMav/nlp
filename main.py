import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import torch
import transformers as ppb
import warnings

from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data = pd.read_csv("nlp/labeled.csv")
batch = data[:2000]

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel,
    ppb.DistilBertTokenizer,
    'distilbert-base-uncased'
)

## Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


tokenized = batch['comment'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
model.eval()
#with torch.no_grad():
#    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# Перемещение модели и тензоров на GPU (если доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Инференс на GPU или CPU
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()
labels = batch['toxic']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, Jtest_size=0.3, random_state=42)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

y_pred = clf.predict(test_features)

with open('result.txt', 'w') as f:
    f.write(classification_report(test_labels, y_pred))
