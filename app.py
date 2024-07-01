from flask import Flask, render_template, request, url_for
import os
import re
import string
import pandas as pd
import pickle
import torch
import torch.nn as nn
import transformers
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model_name = "bert-base-cased"
transformer = AutoModel.from_pretrained(model_name)

class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) # Flatten
        self.logit = nn.Linear(hidden_sizes[-1], num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs

discriminator = Discriminator(input_size=768, hidden_sizes=[768], num_labels=2, dropout_rate=0.1)

# Load the checkpoint on CPU and extract model state dictionary
checkpoint_path = 'discriminator_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    discriminator.load_state_dict(checkpoint['model_state_dict'])

def predict(text, tokenizer, transformer, discriminator):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=100)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation

    with torch.no_grad():
        last_rep, logits, probs = discriminator(hidden_states)
        _, preds = torch.max(probs, dim=1)
    
    return preds, probs

text = "Your input text here"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("textmodel.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model():
    return render_template("textmodel.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'fileUpload' in request.files:
        file = request.files['fileUpload']
        if file.filename != '':
            file.save(os.path.join('data', file.filename))
            return render_template('response.html')
    return "No CSV file uploaded!"

@app.route('/upload_text', methods=['POST'])
def pre():
    if request.method == 'POST':
        txt = request.form['txt']
        preds, probs = predict(txt, bert_tokenizer, transformer, discriminator)
        print(f"Predicted class: {preds.item()}, Probabilities: {probs}")
        return render_template("textmodel.html", result={'preds': preds.item(), 'probs': probs.tolist()})
    else:
        return '' 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
