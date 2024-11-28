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
data_path = './cifar-100'

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

# Residual block 구조 정의
class BasicBlock(nn.Module):
    mul=1
    def __init__(self, in_planes, out_planes, stride = 1):
        super(BasicBlock, self).__init__()

        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias= False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로 너비와 높이는 항시 유지됨.
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함.
        self.shortcut == nn.Sequential()

        # 만약 size가 안 맞아 합 연산이 불가하다면, 연산이 가능하도록 모양을 맞춰 줌
        if stride != 1: # x와(
            self.shortcut == nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x) # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out

class BottleNeck(nn.Module):
    mul = 4
    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()

        #첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_planes*self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes*self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes*self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes*self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes*self.mul)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(MyModel, self).__init__()
        #RGB 3개채널에서 64개의 Kernel 사용
        self.in_planes = 64

        # Resnet 논문 구조 그대로 구현
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512 * block.mul, num_classes)

    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out


def ResNet18():
    return MyModel(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return MyModel(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return MyModel(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return MyModel(BottleNeck, [3, 4, 23, 3])

def ResNet152():
    return MyModel(BottleNeck, [3, 8, 36, 3])

# Simple Learning Rate Scheduler
def lr_scheduler(optimizer, epoch):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Xavier
def init_weights(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# 모델 인스턴스를 생성
model = ResNet152()
# ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 중에 택일하여 사용
model.apply(init_weights)

# 모델을 선택한 디바이스(CPU 또는 GPU)로 이동
model = model.to(device)

learning_rate = 1e-4
# 모델 파라미터를 업데이트할 옵티마이저 정의 (여기서는 SGD 사용)
#optimizer = optim.SGD(model.parameters(), lr=1e-1)  # 학습률(lr)을 0.0001로 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


max_epoch = 150  # 최대 학습 에폭 수 설정

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
