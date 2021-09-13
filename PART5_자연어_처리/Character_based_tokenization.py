# 영어 unicode
print([chr(k) for k in range(65, 91)]) # 영어 대문자
print([chr(k) for k in range(97, 123)]) # 영어 소문자

# 특수 문자 및 숫자 unicode
print([chr(k) for k in range(32, 48)])
print([chr(k) for k in range(58, 65)])
print([chr(k) for k in range(91, 97)])
print([chr(k) for k in range(123, 127)])
print([chr(k) for k in range(48, 58)]) # 숫자

# 한국어 unicode
print([chr(k) for k in range(int('0xAC00', 16), int('0xD7A3', 16) + 1)]) # 모든 완성형 한글 11,172자
print([chr(k) for k in range(int('0x3131', 16), int('0x3163', 16) + 1)]) # 자모

idx2char = {0:'<pad>', 1:'<unk>'}

srt_idx = len(idx2char)
for x in range(32, 127):
    idx2char.update({srt_idx: chr(x)})
    srt_idx += 1

# 한글 추가는 밑의 코드를 실행합니다.
for x in range(int('0x3131', 16), int('0x3163', 16) + 1):
    idx2char.update({srt_idx: chr(x)})
    srt_idx += 1

for x in range(int('0xAC00', 16), int('0xD7A3', 16) + 1):
    idx2char.update({srt_idx: chr(x)})
    srt_idx += 1

char2idx = {v:k for k, v in idx2char.items()}
print([char2idx.get(c, 0) for c in '그래서 Jason에게 사과를 했다'])
print([char2idx.get(c, 0) for c in 'ㅇㅋ! ㄱㅅㄱㅅ'])