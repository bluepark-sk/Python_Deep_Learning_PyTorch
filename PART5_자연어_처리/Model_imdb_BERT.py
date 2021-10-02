import sys
import re
import random
import torch
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(len(tokenizer.vocab))

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
print(max_input_length)

# New Tokenizer
def new_tokenizer(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2] # [CLS], [SEP] Special Token 2자리 미리 빼두기
    return tokens

# Data Cleansing
def PreProcessingText(input_sentence):
    input_sentence = input_sentence.lower()
    input_sentence = re.sub('<[^>]*>', repl=' ', string=input_sentence) # '<br />' 처리
    input_sentence = re.sub('[!"#%&\\()*+,-./:;<=>?@[\\\\]^_`{\}~]', repl=' ', string=input_sentence) # 특수 문자 처리
    input_sentence = re.sub('\\s+', repl=' ', string=input_sentence) # 연속된 띄어쓰기 처리
    if input_sentence:
        return input_sentence

def PreProc(list_sentence):
    return [tokenizer.convert_tokens_to_ids(PreProcessingText(x)) for x in list_sentence]

# Data Setting
TEXT = data.Field(batch_first=True,
                  use_vocab=False,
                  tokenize=new_tokenizer,
                  preprocessing=PreProc,
                  init_token=tokenizer.cls_token_id,
                  eos_token=tokenizer.sep_token_id,
                  pad_token=tokenizer.pad_token_id,
                  unk_token=tokenizer.unk_token_id)

LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)

# Making vocab
LABEL.build_vocab(train_data)

# Splitting Valid set
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

model_config = {'batch_size': 10}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(datasets=(train_data, valid_data, test_data),
                                                                           batch_size=model_config['batch_size'],
                                                                           device=device)

# Model
bert = BertModel.from_pretrained('bert-base-uncased')
model_config['emb_dim'] = bert.config.to_dict()['hidden_size']
model_config['output_dim'] = 1
print(model_config['emb_dim'])

class SentenceClassification(nn.Module):
    def __init__(self, **model_config):
        super(SentenceClassification, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(model_config['emb_dim'], model_config['output_dim'])
    
    def forward(self, x):
        pooled_cls_output = self.bert(x)[1] # [CLS] Pooled_output 사용
        return self.fc(pooled_cls_output)

# Accuracy Function
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Train Function
def train(model, iterator, optimizer, loss_fn, idx_Epoch, **model_params):
    Epoch_loss = 0
    Epoch_acc = 0

    model.train()
    batch_size = model_params['batch_size']

    for idx, batch in enumerate(iterator):
        # Initializing
        optimizer.zero_grad()

        # Forward
        predictions = model(batch.text).squeeze()
        loss = loss_fn(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        sys.stdout.write('\r' + f'[Train] Epoch: {idx_Epoch:^3}'\
            f'[{(idx + 1) * batch_size} / {len(iterator) * batch_size}({100. * (idx + 1) / len(iterator) :.4}%)]'\
            f'  Loss: {loss.item():.4}'\
            f'  Acc: {acc.item():.4}')
        
        # Backward
        loss.backward()
        optimizer.step()

        # Update Epoch Performance
        Epoch_loss += loss.item()
        Epoch_acc += acc.item()

    return Epoch_loss/len(iterator), Epoch_acc/len(iterator)

# Evaluate Function
def evaluate(model, iterator, loss_fn):
    Epoch_loss = 0
    Epoch_acc = 0

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = loss_fn(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            Epoch_loss += loss.item()
            Epoch_acc += acc.item()

    return Epoch_loss / len(iterator), Epoch_acc / len(iterator)

model = SentenceClassification(**model_config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.BCEWithLogitsLoss().to(device)
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# count_parameters(model) # 109483009

# Train
N_EPOCH = 5

best_valid_loss = float('inf')
model_name = 'BERT'

print('----------------------------------------')
print(f'Model name: {model_name}')
print('----------------------------------------')

for Epoch in range(N_EPOCH):
    train_loss, train_acc = train(model, train_iterator, optimizer, loss_fn, Epoch, **model_config)
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_fn)
    print('')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'./{model_name}.pt')
        print(f'\t Saved at {Epoch}-Epoch')

    print(f'\t Epoch: {Epoch} | Train Loss: {train_loss:.4} | Train Acc: {train_acc:.4}')
    print(f'\t Epoch: {Epoch} | Valid Loss: {valid_loss:.4} | Valid Acc: {valid_acc:.4}')

# Test set
model.load_state_dict(torch.load(f'./{model_name}.pt'))
test_loss, test_acc = evaluate(model, test_iterator, loss_fn)
print(f'Test Loss: {test_loss:.4} | Test Acc: {test_acc:.4}')