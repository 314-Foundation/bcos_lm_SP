import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

EPOCHS = 10

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
from transformers import BertForSequenceClassification

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from datasets import load_dataset
dataset = load_dataset("ag_news")

def to_sentiment(rating):
    rating = str(rating)
    if rating == 'positive':
        return 0
    else: 
        return 1

#df['sentiment_score'] = df.sentiment.apply(to_sentiment)
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
sample_txt = 'I want to learn how to do sentiment analysis using BERT and tokenizer.'

encoding = tokenizer.encode_plus(sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
  truncation = True
)

encoding.keys()

token_lens = []
token_lens_ = []

for txt in dataset['train']['text']:
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens_.append(len(tokens))


MAX_LEN = 400


class AGNewsDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.texts)
  
    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
           text,
           add_special_tokens=True,
           max_length=self.max_len,
           return_token_type_ids=False,
           pad_to_max_length=True,
           return_attention_mask=True,
           return_tensors='pt',
           truncation = True
        )

        return {
           'text': text,
           'input_ids': encoding['input_ids'].flatten(),
           'attention_mask': encoding['attention_mask'].flatten(),
           'targets': torch.tensor(target, dtype=torch.long)
        }


class MovieReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
           review,
           add_special_tokens=True,
           max_length=self.max_len,
           return_token_type_ids=False,
           pad_to_max_length=True,
           return_attention_mask=True,
           return_tensors='pt',
           truncation = True
        )

        return {
           'review_text': review,
           'input_ids': encoding['input_ids'].flatten(),
           'attention_mask': encoding['attention_mask'].flatten(),
           'targets': torch.tensor(target, dtype=torch.long)
        }
     

#df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
#df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

#df_train.shape, df_val.shape, df_test.shape

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = MovieReviewDataset(
        reviews=df.review.to_numpy(),
        targets=df.sentiment_score.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_data_loader_(data, tokenizer, max_len, batch_size):
    ds = AGNewsDataset(
        texts=data['text'],
        targets=data['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=1
    )


BATCH_SIZE = 16

#train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
#val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
#test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

train_data_loader_ = create_data_loader_(dataset['train'], tokenizer, MAX_LEN, BATCH_SIZE)
#val_data_loader_ = create_data_loader_(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader_ = create_data_loader_(dataset['test'], tokenizer, MAX_LEN, BATCH_SIZE)

#data = next(iter(train_data_loader))
data_ = next(iter(train_data_loader_))

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

last_hidden_state, pooled_output = bert_model(
    input_ids=encoding['input_ids'], 
    attention_mask=encoding['attention_mask']
)


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):

        out_bert = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )

        #import pdb
        #pdb.set_trace()

        #output = self.drop(pooled_output)
        output = self.drop(out_bert[1])
        return self.out(output)


import torch
torch.cuda.empty_cache()

class_names = ['negative', 'positive']

#model = SentimentClassifier(len(class_names))
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=4, #len(class_names),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


model = model.to(device)     

input_ids_ = data_['input_ids'].to(device)
attention_mask_ = data_['attention_mask'].to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader_) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        #import pdb
        #pdb.set_trace()

        _, preds = torch.max(outputs[0], dim=1)

        #import pdb
        #pdb.set_trace()

        loss = loss_fn(outputs[0], targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs[0], dim=1)

            loss = loss_fn(outputs[0], targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader_, loss_fn, optimizer, device, scheduler, 120000) #len(df_train))

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, test_data_loader_, loss_fn, device, 7600) #len(df_val))

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state__.bin')
        best_accuracy = val_acc


test_acc, _ = eval_model(model, test_data_loader_, loss_fn, device, 7600) #len(df_test))

test_acc.item()

def get_predictions(model, data_loader):
    model = model.eval()
  
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs[0], dim=1)

            probs = F.softmax(outputs[0], dim=1)
 
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)
     

def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
