S1 = '나는 책상 위에 사과를 먹었다'

print([S1[i:i+1] for i in range(len(S1))]) # uni-gram
print([S1[i:i+2] for i in range(len(S1))]) # bi-gram
print([S1[i:i+3] for i in range(len(S1))]) # tri-gram