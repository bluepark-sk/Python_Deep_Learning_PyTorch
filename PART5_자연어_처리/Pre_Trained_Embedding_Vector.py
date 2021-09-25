import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

# Data Setting
TEXT = data.Field(batch_first=True,
                  fix_length=500,
                  tokenize=str.split,
                  pad_first=True,
                  pad_token='[PAD]',
                  unk_token='[UNK]')

LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)

# Data Length
print(f'Train Data Length: {len(train_data.examples)}')
print(f'Test Data Length: {len(test_data.examples)}')

# Data Fields
print(train_data.fields)
print(test_data.fields)

# Data Sample
print('---- Data Sample ----')
print('Input: ')
print(' '.join(vars(train_data.examples[1])['text']), '\\n')
print('Label: ')
print(vars(train_data.examples[1])['label'])

# Data Cleansing
import re

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

# Token Vocabulary (Pre-trained Vectors)
TEXT.build_vocab(train_data,
                 min_freq=2,
                 max_size=None,
                 vectors='glove.6B.300d')

LABEL.build_vocab(train_data)

# Vocabulary Info
print(f'Vocab Size: {len(TEXT.vocab)}')

print('Vocab Examples: ')
for idx, (k, v) in enumerate(TEXT.vocab.stoi.items()):
    if idx >= 10:
        break
    print('\\t', k, v)

print('----------------------------------------')

# Label Info
print(f'Label Size: {len(LABEL.vocab)}')

print('Label Examples: ')
for idx, (k, v) in enumerate(LABEL.vocab.stoi.items()):
    print('\\t', k, v)

# Check embedding vectors
print(TEXT.vocab.vectors.shape)

# Splitting Valid set
import random

train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(datasets=(train_data, valid_data, test_data), batch_size=30, device=device)