import sys
import re
import random
import torch
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy import datasets

# Data Setting
TEXT = data.Field(batch_first=True,
                  fix_length=500,
                  tokenize=str.split,
                  pad_first=True,
                  pad_token='<pad>',
                  unk_token='<unk>')

LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)

# Data Cleansing
def PreProcessingText(input_sentence):
    input_sentence = input_sentence.lower()
    input_sentence = re.sub('<[^>]*>', repl=' ', string=input_sentence) # '<br />' 처리
    input_sentence = re.sub('[!"#%&\\()*+,-./:;<=>?@[\\\\]^_`{\}~]', repl=' ', string=input_sentence) # 특수 문자 처리
    input_sentence = re.sub('\\s+', repl=' ', string=input_sentence) # 연속된 띄어쓰기 처리
    if input_sentence:
        return input_sentence

for example in train_data.examples:
    vars(example)['text'] = PreProcessingText(' '.join(vars(example)['text'])).split()

for example in test_data.examples:
    vars(example)['text'] = PreProcessingText(' '.join(vars(example)['text'])).split()

model_config = {'emb_type': 'glove', 'emb_dim': 300}

# Making vocab
TEXT.build_vocab(train_data,
                 min_freq=2,
                 max_size=None,
                 vectors=f'glove.6B.{model_config["emb_dim"]}d')

LABEL.build_vocab(train_data)

model_config['vocab_size'] = len(TEXT.vocab)

# Splitting Valid set
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

model_config['batch_size'] = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(datasets=(train_data, valid_data, test_data),
                                                                           batch_size=model_config['batch_size'],
                                                                           device=device)

# Check batch data
sample_for_check = next(iter(train_iterator))
print(sample_for_check)
print(sample_for_check.text)
print(sample_for_check.label)

# Model
class SentenceClassification(nn.Module):
    def __init__(self, **model_config):
        super(SentenceClassification, self).__init__()

        if model_config['emb_type'] == 'glove' or 'fasttext':
            self.emb = nn.Embedding(model_config['vocab_size'], model_config['emb_dim'], _weight=TEXT.vocab.vectors)
        else:
            self.emb = nn.Embedding(model_config['vocab_size'], model_config['emb_dim'])
        
        self.bidirectional = model_config['bidirectional']
        self.num_direction = 2 if model_config['bidirectional'] else 1
        self.model_type = model_config['model_type']

        self.RNN = nn.RNN(input_size=model_config['emb_dim'],
                          hidden_size=model_config['hidden_dim'],
                          dropout=model_config['dropout'],
                          bidirectional=model_config['bidirectional'],
                          batch_first=model_config['batch_first'])
        
        self.LSTM = nn.LSTM(input_size=model_config['emb_dim'],
                            hidden_size=model_config['hidden_dim'],
                            dropout=model_config['dropout'],
                            bidirectional=model_config['bidirectional'],
                            batch_first=model_config['batch_first'])
        
        self.GRU = nn.GRU(input_size=model_config['emb_dim'],
                          hidden_size=model_config['hidden_dim'],
                          dropout=model_config['dropout'],
                          bidirectional=model_config['bidirectional'],
                          batch_first=model_config['batch_first'])
        
        self.fc = nn.Linear(model_config['hidden_dim']*self.num_direction, model_config['output_dim'])
        self.drop = nn.Dropout(model_config['dropout'])
    
    def forward(self, x): # x : (Batch_Size, Max_Seq_Length)
        emb = self.emb(x) # emb : (Batch_Size, Max_Seq_Length, Emb_dim)

        if self.model_type == 'RNN':
            output, hidden = self.RNN(emb)
        elif self.model_type == 'LSTM':
            output, (hidden, cell) = self.LSTM(emb)
        elif self.model_type == 'GRU':
            output, hidden = self.GRU(emb)
        else:
            raise NameError('Select model_type in [RNN, LSTM, GRU]')
        
        # output : (Batch_Size, Max_Seq_Length, Hidden_dim * num_direction)
        # hidden : (num_direction, Batch_Size, Hidden_dim)

        last_output = output[:, -1, :] # last_output : (Batch_Size, Hidden_dim * num_direction)

        return self.fc(self.drop(last_output))

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

model_config.update(dict(batch_first=True,
                         model_type='RNN',
                         bidirectional=True,
                         hidden_dim=128,
                         output_dim=1,
                         dropout=0))

model = SentenceClassification(**model_config).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss().to(device)

# Train
N_EPOCH = 5

best_valid_loss = float('inf')
model_name = f"{'bi-' if model_config['bidirectional'] else ''}{model_config['model_type']}_{model_config['emb_type']}"

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

model_config['model_type'] = 'RNN'
model = SentenceClassification(**model_config).to(device)
model.load_state_dict(torch.load(f"./{'bi-' if model_config['bidirectional'] else ''}{model_config['model_type']}_{model_config['emb_type']}.pt"))

def predict_sentiment(model, sentence):
    model.eval()
    indexed = TEXT.numericalize(TEXT.pad([TEXT.tokenize(PreProcessingText(sentence))]))
    input_data = torch.LongTensor(indexed).to(device)
    prediction = torch.sigmoid(model(input_data))
    return prediction.item()

test_sentence = 'this movie is FUN'
predict_sentiment(model=model, sentnece=test_sentence)