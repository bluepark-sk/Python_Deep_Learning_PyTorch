# GAN (generative Adversarial Networks)
- GAN은 데이터를 만들어내는 Generator와 만들어진 데이터를 평가하는 Discriminator가 서로 대립(Adversarial)적으로 학습해가며 성능을 점차 개선해 나가자는 개념으로, 예측의 목적을 뛰어 넘어 생성을 목적으로 함
- GAN의 최종 목적은 데이터를 '생성'해내는 것이기 때문에 Generator를 학습 시키는 것
- 지폐위조범(Generator) / 경찰(Discriminator)
    - 속이고 구별하는 서로의 능력이 발전하고 결과적으로 진짜 지폐와 위조 지폐를 구별할 수 없을 정도에 이름  
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FA1Gzp%2Fbtq1KcZZheV%2FNaluIT2J6A9JumRLbACC40%2Fimg.png" width="60%" height="60%">

***

- Value Function
    - $minmaxV(D,G) = E_{x ~ P_{data(X)}}[logD(X)] + E_{z ~ P_{z(Z)}}[log(1-D(G(z)))]$
    - Discriminator
        - Maximizing V
        - 진짜 데이터를 1로 구분하고 가짜 데이터를 0으로 구분할 때 V는 최댓값을 가짐
    - Generator
        - Minimizing V
        - 가짜 데이터를 1로 속일 때 V는 최솟값을 가짐

***

- **학습 과정**
    - Discriminator
        - Input Noise를 G에 넣어 만들어진 가짜 데이터와 진짜 데이터를 각각 D에 넣어 가짜는 0, 진짜는 1로 label을 설정해 학습 진행
            - D는 진짜 데이터를 진짜로, 가짜 데이터를 가짜로 분류하는 것이 목표이기 대문에 가짜, 진짜 데이터가 모두 필요
    - Generator
        - Input Noise로부터 G를 통해 가짜 데이터를 만들고 이를 D에 Input으로 넣음
        - G의 Output을 다시 D의 Input으로 넣어 계산
            - G의 목적이 D를 속이는 것이기 때문에 D의 에러를 통해 G의 Back Propagation을 계산 (G는 D를 속이는 방향으로 G의 Weight을 업데이트)
            - 이 과정에서 D의 Weight은 업데이트 하지 않음. D는 단지 G의 Weight을 구하기 위해 에러를 전파시키기만 할 뿐. Weight Freezing.
    - D, G를 번갈아 가면서 학습 진행을 반복하면 D의 입장에서 진짜, 가짜 구분 확률이 0.5에 수렴

***

- GAN의 단점
    - Mode Collapse : G의 목적은 D를 속이는 것인데, D가 속기만 하면 된다는 것으로도 해석 가능. 즉, G는 D가 속기만 하면 되는 특정 데이터만 만드려고 노력

***

- 다양한 GAN 모델
    - BigGAN
    - CycleGAN
    - Deep Photo Style Transfer
    - Style Transfer for Anime Sketches
    - StarGAN
    - CAN
    - SRGAN
    - Globally and Locally Consistent Image Completion

<br/>
<br/>

# 강화학습 (Reinforcement Learning)
- 강화학습은 현재 상태(State)에서 어떤 행동(Action)을 취해야 먼 미래에 보상(Reward)을 최대로 받을지를 학습
- 강화학습의 기초가 되는 알고리즘으로 Q-learning, SARSA 알고리즘 등이 있으며, 딥러닝과 결합하면서 Deep Reinforcement Learning이 개발됨
- 강화학습을 위해서는 수많은 Episode(Current State, Action, Reward, Next State)가 필요하다 보니 주로 Game 환경에서 개발됨

***

- **학습 과정**
    - Start 지점에서 Goal로 최단거리로 가도록 하고 싶을 때 다음으로 취할 행동을 Action, 현재의 위치(1, 1)를 State 라고 함
    - 각 Action이 좋은 행동인지, 아닌지 판단할 수 있는 Feedback이 필요하고 이 Feedback을 Reward 라고 정의. 강화학습은 Reward를 통해 현재 State에서 어떤 Action을 취하는 것이 좋은지를 측정하는 값 (Q-Value) 학습
    - Goal에 대해 각 State에서 Action에 대한 최적의 Q-Value 학습, 이는 현재 State에서 미래 보상의 합이 최대가 되도록 학습
        - 우리의 목적은 Goal에 갈 수 있는 가장 좋은 Action을 취하는 것. 즉, 지금 당장 좋은 것도 좋은 것이지만 궁극적으로는 미래의 목표를 달성할 수 있는 Action을 선택
        - 먼 미래에 더 좋은 상황을 가져오는 Action을 선택하도록 하기 위해 Discount Factor(할인율) 도입
        - 각 State가 받는 보상 = 현재 Action을 통해 받는 Reward + 미래 보상(할인율)의 합

***

- Value Function
    - $V^\pi(s_{t}) = r_{t} + \gamma*r_{t+1} + \gamma^2*r_{t+2} + ... = \sum_{i=0}^\infty\gamma^i*r_{t+i}$
    - $\pi^* = argmax_{\pi}V^\pi(s)$ for all s
    - Q-Learning
        - $\hat{Q}(s, a) = r(s, a) + \gamma * max_{a^\prime}\hat{Q}(s^\prime, a^\prime)$
        - 현재 State의 Action에 대한 Q-Value 값은 즉시 Reward + 다음 State에서의 가장 큰 q값 * Discount Factor로 업데이트  
    ![Q-Learning](https://blog.floydhub.com/content/images/2019/05/image-25.png)

<br/>
<br/>

# Domain Adaptation
- 특정 도메인 내에 있는 데이터가 부족할 때, 특정 도메인과 매우 유사한 도메인 정보를 이용해 문제를 해결
- 우리가 풀어야 할 도메인을 Target Domain, 이용할 수 있는 비슷한 도메인을 Source Domain이라 할 때 레이블 정보가 포함되어 있는 대규모의 Source Domain의 데이터를 이용해 Classifier를 학습

<br/>
<br/>

# Continual Learning
- Catastrophic Forgetting
    - 특정 데이터로 이미 학습된 모델이 새로운 데이터를 바탕으로 추가로 학습을 진행했을 때 과거에 학습된 데이터에 대한 능력이 사라짐
    - 이 Catastrophic Forgetting을 최소화하고자 하는 학습 방식이 Continual Learning
    - 기존에 학습된 모델에 새로운 클래스를 추가로 학습하되, 기존에 학습된 능력을 잊지 않게 하고 싶은 것

- **방법론**
    - Selective Retraining
        - 동일한 모델 구조에 대해 (t-1) 시점까지 Task1에 대해 학습한 모델의 파라미터와 t 시점에서 Task2에 대해 학습한 모델의 파라미터 간 관계가 높은 파라미터를 추출해 다시 학습을 진행
    - Dynamic Network Expansion
        - Selective Retraining에서 학습이 잘 안 될 때, 파라미터 값이 비교적 작은 값으로 연결된 노드를 제거하고 모델의 크기를 늘림
    - Network Split/Duplication
        - 기존 노드의 값이 크게 변했을 때 Catastrophic Forgetting 현상이 발생할 수 있기 때문에 기존 노드 값을 복사해 모델 구조에 변화를 줌

<br/>
<br/>

# Object Detection
- 어떤 이미지나 비디오 자체를 특정 클래스로 분류하는 Classification Task를 수행함과 동시에 해당 클래스가 이미지나 비디오 내 어느 위치에 있는지 위치정보까지 예측
- 위치 정보
    - 물체의 특정 지점 (x, y) 좌표
    - 특정 지점으로부터 가로, 세로 길이 (width, height)
- 예측된 (x, y, w, h)를 통해 물체에 대한 네모를 그릴 수 있으며, 이 네모를 Boundary Box 라고 지칭
- CNN 기반  
<img src="https://www.researchgate.net/profile/Kedar-Potdar-2/publication/329217107/figure/fig4/AS:697578261852162@1543327026650/Object-detection-in-a-dense-scene.ppm" width="60%" height="60%">

***

- Object Detection
    - Regional Proposal
        - 이미지나 비디오 내에서 물체의 위치정보 추출
    - Classification
        - 추출한 위치 정보의 물체를 분류
    - 2 Stage
        - 다른 알고리즘을 이용해 Regional Proposal 진행 후 Classification 처리
        - 정확도가 1 Stage에 비해 높음
        - Regional Convolutional Neural Network (RCNN)
    - 1 Stage
        - Regional Proposal과 Classification 단계를 하나의 딥러닝 모델로 동시 처리 (end-to-end 방식)
        - 처리 속도가 2 Stage에 비해 빠름
        - 물체를 탐지하는 속도가 실제 산업에서 매우 중요하기 때문에 1 Stage 방식으로 많이 연구되고 있음
        - You only look once (YOLO)

<br/>
<br/>

# Segmentation
- Segmentation은 특정 위치에 Boundary Box로 물체 존재 유무를 표현하는 것의 한계를 극복하기 위해 이미지 및 비디오 내에 있는 모든 픽셀에 대해 특정 클래스로 예측  
![Segmentation](https://blog.kakaocdn.net/dn/CRvWU/btqSsQypZlk/gAMakLhRAykcULSIsSaP60/img.png)

<br/>
<br/>

# Meta Learning
- Meata Learning은 학습하는 방법을 학습하는 것을 의미
- 다양한 Task에 대해 각 Task 별로 소량의 데이터를 이용해 어느 정도 학습 효과를 나타낼 수 있는 방법론을 연구

***

- **대표 방법론**
    - Weight Initialization Point 찾기
        - 학습이 빠르게 되는 초기 파라미터 분포를 찾는 것
        - Gradient Descent를 이용해 Back Propagation을 진행할 때, 최소한의 업데이트로 최소의 Loss 값을 가질 수 있도록 하는 것이 목표  
        <img src="https://miro.medium.com/max/1749/1*RNjfqTZXzQE7S5tV7Do_0Q.png" width="50%" height="50%">
        - 각 Task1, Task2, Task3의 최소 Loss 값을 갖는 파라미터 값을 각각 $\theta_1^*, \theta_2^*, \theta_3^*$ 라고 했을 때, $\theta_1^*, \theta_2^*, \theta_3^*$ 로 쉽게 이동할 수 있는 Point로 Meta-Learning의 값을 업데이트
        - 업데이트를 통해 최종적으로 완성된 값을 특정 Task를 수행할 때 시작되는 Weight Initialization 지점으로 설정하면 각 Task1, Task2, Task3에서 쉽게 파라미터를 업데이트 할 수 있음

<br/>
<br/>

# AutoML
- 딥러닝 모델의 레이어 구성 및 노드 수 등 하이퍼파라미터를 사용자가 설정해서 모델을 설계하는 대신, 컴퓨터가 스스로 모델을 설계하는 것

***

- **대표 방법론**
    - Auto Augmentation
        - Data Augmentation 기법을 자동으로 설계
    - Neural Architecture Search
        - 특정 Task를 풀기 위해 딥러닝 모델 구조를 자동으로 설계