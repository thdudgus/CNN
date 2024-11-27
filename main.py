import os
import csv
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

print(torch.__version__)

# CIFAR-100 데이터셋 디렉토리 경로 설정
data_path = './cifar100'

# 학습 및 테스트를 위한 배치 크기 정의
batch_size = 64

# 학습 데이터에 적용할 변환(전처리) 정의
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 이미지를 무작위로 좌우 반전
    transforms.RandomCrop(32, padding=4),  # 이미지를 무작위로 잘라내되, 가장자리 여백 추가
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 이미지 정규화
])
# 테스트 데이터에 적용할 변환(전처리) 정의
transform_test = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 이미지 정규화
])

# CIFAR-100 학습 데이터셋 로드 및 변환 적용
train_dataset = torchvision.datasets.CIFAR100(data_path, train=True, transform=transform_train, download=True)
# CIFAR-100 테스트 데이터셋 로드 및 변환 적용
test_dataset = torchvision.datasets.CIFAR100(data_path, train=False, transform=transform_test, download=True)

# 학습 데이터셋을 위한 DataLoader 생성 (배치 처리 및 데이터 순서 섞기)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 테스트 데이터셋을 위한 DataLoader 생성 (배치 처리만, 데이터 순서 섞지 않음)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.is_available()
print(torch.cuda.is_available())  # True면 CUDA 사용 가능
print('Current Device : {}'.format(device))

# 학습 데이터 로더에서 한 번에 하나의 배치를 가져옴
sample = next(iter(train_dataloader))

# 샘플 배치의 입력 이미지들의 크기(shape)를 출력
print(sample[0].shape)
# 샘플 배치에 해당하는 이미지들의 레이블을 출력
print(sample[1])

fig, ax = plt.subplots(1, 10, figsize=(15, 4))  # 1행 10열의 서브플롯 생성, 전체 크기 설정

# 배치에서 첫 10개의 샘플에 대해 반복
for plot_idx in range(10):
    # 현재 샘플의 이미지를 표시
    # permute(1, 2, 0)을 사용하여 차원 순서를 (C, H, W)에서 (H, W, C)로 변경
    ax[plot_idx].imshow(sample[0][plot_idx].permute(1, 2, 0))

    # 해당 이미지의 레이블을 제목으로 설정
    ax[plot_idx].set_title('LABEL : {}'.format(sample[1][plot_idx]))

    # x축과 y축의 눈금을 제거
    ax[plot_idx].set_xticks([])
    ax[plot_idx].set_yticks([])

# 플롯 표시
plt.show()


class MyModel(nn.Module):
    def __init__(self, img_size=32, num_class=100):
        super(MyModel, self).__init__()

        self.img_size = img_size
        self.num_class = num_class

        # Convolutional Layers
        self.features = nn.Sequential(
            # Conv Layer Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce image size by half

            # Conv Layer Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce image size by half

            # Conv Layer Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce image size by half
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),  # 4x4은 이미지 크기 감소
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class)  # CIFAR-100은 100개의 클래스
        )

    def forward(self, img):
        # Feature extraction
        features = self.features(img)  # Conv layers 통과
        features = features.view(features.size(0), -1)  # Flatten
        # Classification
        out = self.classifier(features)  # Fully connected layers 통과
        return out


# 모델 인스턴스를 생성
model = MyModel()
# 모델을 선택한 디바이스(CPU 또는 GPU)로 이동
model = model.to(device)

# 모델 파라미터를 업데이트할 옵티마이저 정의 (여기서는 SGD 사용)
optimizer = optim.SGD(model.parameters(), lr=1e-1)  # 학습률(lr)을 0.0001로 설정

# 모델 아키텍처 출력
print(model)



def train(model, optimizer, sample):
    # 모델을 학습 모드로 설정
    model.train()

    # 손실 함수 정의 및 옵티마이저의 기울기 초기화
    criterion = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수 사용
    optimizer.zero_grad()

    # 샘플에서 입력 이미지와 레이블 추출
    img = sample[0].float().to(device)  # 입력 이미지를 float 형식으로 변환하고 디바이스로 이동
    label = sample[1].long().to(device)  # 레이블을 long 형식으로 변환하고 디바이스로 이동

    # 순전파(forward pass): 입력 데이터를 모델에 전달하여 예측 값 계산
    pred = model(img)  # 모델을 통해 예측 값 계산
    num_correct = sum(torch.argmax(pred, dim=-1) == label)  # 예측 값 중 정답과 일치하는 개수 계산

    # 손실 계산, 기울기 계산, 가중치 업데이트
    pred_loss = criterion(pred, label)  # 예측 값과 실제 레이블 간의 손실 계산
    pred_loss.backward()  # 역전파(backward pass)를 통해 기울기 계산
    optimizer.step()  # 옵티마이저를 통해 모델의 파라미터 업데이트

    # 손실 값과 정확한 예측의 개수를 반환
    return pred_loss.item(), num_correct.item()



def test(model, sample):
    # 모델을 평가 모드로 설정
    model.eval()

    # 손실 계산을 위한 기준 정의
    criterion = nn.CrossEntropyLoss()

    # 테스트 시에는 기울기 계산을 비활성화 (추론 모드)
    with torch.no_grad():
        # 샘플에서 이미지와 레이블 추출
        img = sample[0].float().to(device)  # 입력 이미지를 float 형식으로 변환하고 디바이스로 이동
        label = sample[1].long().to(device)  # 레이블을 long 형식으로 변환하고 디바이스로 이동

        # 순전파(forward pass) 수행하여 모델 예측 값 계산
        pred = model(img)  # 모델을 통해 예측 값 계산
        pred_loss = criterion(pred, label)  # 예측 값과 실제 레이블 간의 손실 계산

        # 예측 값 중 정답과 일치하는 샘플 개수 계산
        num_correct = sum(torch.argmax(pred, dim=-1) == label)

    # 손실 값과 정확히 예측된 샘플 개수를 반환
    return pred_loss.item(), num_correct.item()


max_epoch = 60  # 최대 학습 에폭 수 설정

# 학습 및 테스트 손실을 저장할 리스트 초기화
tr_loss_saver = []
te_loss_saver = []

# 각 에폭에 대해 반복
for epoch in tqdm(range(max_epoch)):
    ### 학습 단계 (Train Phase)

    # 학습 단계의 손실 및 정확도 초기화
    train_loss = 0.0
    train_accu = 0.0

    # 학습 데이터 로더를 반복
    for idx, sample in enumerate(train_dataloader):
        # 현재 배치에 대해 학습 수행
        curr_loss, num_correct = train(model, optimizer, sample)

        # 총 학습 손실 및 정확도 업데이트
        train_loss += curr_loss / len(train_dataloader)
        train_accu += num_correct / len(train_dataset)

    # 학습 손실 저장
    tr_loss_saver.append(train_loss)

    # 각 에폭 후 모델 상태 저장
    torch.save(model.state_dict(), 'recent.pth')

    ### 테스트 단계 (Test Phase)

    # 테스트 단계의 손실 및 정확도 초기화
    test_loss = 0.0
    test_accu = 0.0

    # 테스트 데이터 로더를 반복
    for idx, sample in enumerate(test_dataloader):
        # 현재 배치에 대해 테스트 수행
        curr_loss, num_correct = test(model, sample)

        # 총 테스트 손실 및 정확도 업데이트
        test_loss += curr_loss / len(test_dataloader)
        test_accu += num_correct / len(test_dataset)

    # 테스트 손실 저장
    te_loss_saver.append(test_loss)

    # 각 에폭별 학습 및 테스트 통계 출력
    print('[EPOCH {}] TR LOSS : {:.03f}, TE LOSS : {:.03f}, TR ACCU: {:.03f}, TE ACCU : {:.03f}'.format(epoch+1, train_loss, test_loss, train_accu, test_accu))



plt.figure(figsize=(4, 3))
plt.plot(tr_loss_saver)  # Plot the training loss
plt.plot(te_loss_saver)  # Plot the testing loss
plt.legend(['Train Loss', 'Test Loss'])  # Add legend to the plot
plt.xlabel('Epoch')  # Label for the x-axis
plt.ylabel('Loss')  # Label for the y-axis
plt.title('Train and Test Loss')  # Title of the plot
plt.show()  # Display the plot





