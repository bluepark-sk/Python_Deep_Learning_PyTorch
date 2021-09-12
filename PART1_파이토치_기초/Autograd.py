import torch
from torch._C import device, dtype

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

BATCH_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10

x = torch.randn(BATCH_SIZE,
                INPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

y = torch.randn(BATCH_SIZE,
                OUTPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

w1 = torch.randn(INPUT_SIZE,
                 HIDDEN_SIZE,
                 device=DEVICE,
                 dtype=torch.float,
                 requires_grad=True)

w2 = torch.randn(HIDDEN_SIZE,
                 OUTPUT_SIZE,
                 device=DEVICE,
                 dtype=torch.float,
                 requires_grad=True)

learning_rate = 1e-6 # hyperparameter
for t in range(1, 501): # epoch : 500
    y_pred = x.mm(w1).clamp(min=0).mm(w2) # activation function for prediction : clamp

    loss = (y_pred - y).pow(2).sum() # loss : 예측값과 실제 레이블 값을 비교해 오차를 계산한 값 (SS)
    if t % 100 == 0:
        print("Iteration: ", t, "\t", "Loss:", loss.item()) # print for monitoring
        # print("w1:", w1, "w2:", w2)
    loss.backward() # 각 파라미터 값에 대해 gradient를 계산하고 이를 통해 Back Propagation 진행

    with torch.no_grad(): # 파라미터 값을 업데이트할 때는 해당 시점의 gradient 값을 고정한 후 업데이트를 진행
        w1 -= learning_rate * w1.grad # 음수를 사용하는 이유 : Loss 값이 최소로 계산될 수 있는 파라미터 값을 찾기 위해 gradient 값에 대한 반대 방향으로 계산
        w2 -= learning_rate * w2.grad
        w1.grad.zero_() # 파라미터 w1 값의 gradient 값을 0으로 초기화
        w2.grad.zero_() # 파라미터 w2 값의 gradient 값을 0으로 초기화