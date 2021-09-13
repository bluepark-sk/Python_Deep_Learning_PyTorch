S1 = '나는 책상 위에 사과를 먹었다'
S2 = '알고 보니 그 사과는 Jason 것이었다'
S3 = '그래서 Jason에게 사과를 했다'

print(S1.split())
print(S2.split())
print(S3.split())

print(list(S1))

token2idx = {}
index = 0
for sentence in [S1, S2, S3]:
    tokens = sentence.split()
    for token in tokens:
        if token2idx.get(token) == None:
            token2idx[token] = index
            index += 1

print(token2idx)

def indexed_sentence(sentence):
    return [token2idx[token] for token in sentence]

S1_i = indexed_sentence(S1.split())
print(S1_i)

S2_i = indexed_sentence(S2.split())
print(S2_i)

S3_i = indexed_sentence(S3.split())
print(S3_i)

# Corpus & Out-of-Vocabulary(OOV)
S4 = '나는 책상 위에 배를 먹었다'
indexed_sentence(S4.split()) # KeyError

# 기존 token 사전에 <unk> token 추가
token2idx = {t: i+1 for t, i in token2idx.items()}
token2idx['<unk>'] = 0

# token이 없을 경우 <unk> token의 0을 치환
def indexed_sentence_unk(sentence):
    return [token2idx.get(token, token2idx['<unk>']) for token in sentence]

indexed_sentence_unk(S4.split())