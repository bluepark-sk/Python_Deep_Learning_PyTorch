# Sentnecepiece
import sentencepiece as spm

s = spm.SentencePieceProcessor(model_file='spm.model')
for n in range(5):
    s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest=-1)

# BERT
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(len(tokenizer.vocab))

sentence = 'My dog is cute. He likes playing'
print(tokenizer.tokenize(sentence))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
print(len(tokenizer.vocab))

sentence = '나는 책상 위에 사과를 먹었다. 알고 보니 그 사과는 Jason 것이었다. 그래서 Jason에게 사과를 했다'
print(tokenizer.tokenize(sentence))