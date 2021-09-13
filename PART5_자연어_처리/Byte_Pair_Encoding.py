# Algorithm 1: Learn BPE operations
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f'Step {i + 1}')
    print(best)
    print(vocab)
    print('\\n')

# 띄어쓰기 vs BPE
S1 = '나는 책상 위에 사과를 먹었다'
S2 = '알고 보니 그 사과는 Jason 것이었다'
S3 = '그래서 Jason에게 사과를 했다'

token_counts = {}
index = 0
for sentence in [S1, S2, S3]:
    tokens = sentence.split()
    for token in tokens:
        if token_counts.get(token) == None:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

token_counts = {' '.join(token):counts for token, counts in token_counts.items()}
print(token_counts)

num_merges = 10
for i in range(num_merges):
    pairs = get_stats(token_counts)
    best = max(pairs, key=pairs.get)
    token_counts = merge_vocab(best, token_counts)
    print(f'Step {i + 1}')
    print(best)
    print(token_counts)
    print('\\n')