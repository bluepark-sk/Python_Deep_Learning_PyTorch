import torch
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy import datasets
from Pre_Trained_Embedding_Vector import TEXT

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

        last_output = output[:, -1, :]
        # -1 : 문장 분류를 위해 사용하고자 하는 모델의 문장 정보는 문장 가장 끝 Token의 RNN Module Ouput 사용
        # last_output : (Batch_Size, Hidden_dim * num_direction)

        return self.fc(self.drop(last_output))

# RNN with Data from Pre_Trained_Embedding_Vector.py
import Pre_Trained_Embedding_Vector

train_iterator = Pre_Trained_Embedding_Vector.train_iterator
device = Pre_Trained_Embedding_Vector.device

sample_for_check = next(iter(train_iterator))
print(sample_for_check)
print(sample_for_check.text)
print(sample_for_check.label)

# model, loss function
model_config = {'emb_type': 'glove', 'emb_dim': 300, 'vocab_size': 95995}
model_config.update(dict(batch_first=True,
                         model_type='RNN',
                         bidirectional=True,
                         hidden_dim=128,
                         output_dim=1, # binary classification
                         dropout=0))

model = SentenceClassification(**model_config).to(device)
loss_fn = nn.BCEWithLogitsLoss().to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

predictions = model.forward(sample_for_check.text).squeeze()
loss = loss_fn(predictions, sample_for_check.label)
acc = binary_accuracy(predictions, sample_for_check.label)

print(predictions)
print(loss, acc)