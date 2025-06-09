import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    """
    기본 CNN 블록
    
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
        kernel_size (int): 컨볼루션 커널 크기
        stride (int): 컨볼루션 스트라이드
        padding (int): 패딩 크기
        pool_size (int): 풀링 크기
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class WiFiCSICNN(nn.Module):
    """
    WiFi CSI 데이터 분류를 위한 CNN 모델
    
    Args:
        num_classes (int): 클래스 수
        num_esp (int): ESP 칩셋 수
    """
    def __init__(self, num_classes=3, num_esp=4):
        super(WiFiCSICNN, self).__init__()
        
        # 각 ESP 칩셋별 특징 추출을 위한 CNN 블록
        self.esp_feature_extractors = nn.ModuleList([
            nn.Sequential(
                CNNBlock(1, 32),  # 입력: (1, 180, 114) -> 출력: (32, 90, 57)
                CNNBlock(32, 64),  # 입력: (32, 90, 57) -> 출력: (64, 45, 28)
                CNNBlock(64, 128)  # 입력: (64, 45, 28) -> 출력: (128, 22, 14)
            ) for _ in range(num_esp)
        ])
        
        # 특징 결합 및 분류를 위한 레이어
        # 각 ESP 특징 추출기의 출력 크기 계산: 128 * 22 * 14 = 39,424
        self.fc1 = nn.Linear(128 * 22 * 14 * num_esp, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 입력 형태: (batch_size, num_esp, time_steps, subcarriers)
        batch_size = x.size(0)
        num_esp = x.size(1)
        
        # 각 ESP 칩셋별 특징 추출
        esp_features = []
        for i in range(num_esp):
            # ESP 데이터 추출 및 채널 차원 추가
            esp_data = x[:, i, :, :].unsqueeze(1)  # (batch_size, 1, time_steps, subcarriers)
            # 특징 추출
            features = self.esp_feature_extractors[i](esp_data)
            # 특징 벡터로 변환
            features = features.view(batch_size, -1)
            esp_features.append(features)
        
        # 모든 ESP 특징 결합
        combined_features = torch.cat(esp_features, dim=1)
        
        # 분류
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class WiFiCSICNNAttention(nn.Module):
    """
    Attention 메커니즘을 적용한 WiFi CSI CNN 모델
    
    Args:
        num_classes (int): 클래스 수
        num_esp (int): ESP 칩셋 수
    """
    def __init__(self, num_classes=3, num_esp=4):
        super(WiFiCSICNNAttention, self).__init__()
        
        # 각 ESP 칩셋별 특징 추출을 위한 CNN 블록
        self.esp_feature_extractors = nn.ModuleList([
            nn.Sequential(
                CNNBlock(1, 32),
                CNNBlock(32, 64),
                CNNBlock(64, 128)
            ) for _ in range(num_esp)
        ])
        
        # 특징 맵 크기
        self.feature_size =  128 * 22 * 14  # 1s: 128 * 7 * 14  #2s: 128 * 15 * 14   #3s: 128 * 22 * 14
        
        # ESP 칩셋 간 Attention 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 분류 레이어
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 입력 형태: (batch_size, num_esp, time_steps, subcarriers)
        batch_size = x.size(0)
        num_esp = x.size(1)
        
        # 각 ESP 칩셋별 특징 추출
        esp_features = []
        for i in range(num_esp):
            esp_data = x[:, i, :, :].unsqueeze(1)
            features = self.esp_feature_extractors[i](esp_data)
            features = features.view(batch_size, -1)
            esp_features.append(features)
        
        # 특징 스택 생성: (batch_size, num_esp, feature_size)
        stacked_features = torch.stack(esp_features, dim=1)
        
        # Attention 가중치 계산
        attention_weights = []
        for i in range(num_esp):
            weight = self.attention(stacked_features[:, i, :])
            attention_weights.append(weight)
        
        attention_weights = torch.cat(attention_weights, dim=1)  # (batch_size, num_esp)
        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(2)  # (batch_size, num_esp, 1)
        
        # 가중 평균 계산
        weighted_features = torch.sum(stacked_features * attention_weights, dim=1)  # (batch_size, feature_size)
        
        # 분류
        x = self.fc1(weighted_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class WiFiCSICNNEnsemble(nn.Module):
    """
    앙상블 방식의 WiFi CSI CNN 모델
    
    Args:
        num_classes (int): 클래스 수
        num_esp (int): ESP 칩셋 수
    """
    def __init__(self, num_classes=3, num_esp=4):
        super(WiFiCSICNNEnsemble, self).__init__()
        
        # 각 ESP 칩셋별 독립적인 CNN 모델
        self.esp_models = nn.ModuleList([
            nn.Sequential(
                CNNBlock(1, 32),
                CNNBlock(32, 64),
                CNNBlock(64, 128),
                nn.Flatten(),
                nn.Linear(128 * 22 * 14, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            ) for _ in range(num_esp)
        ])
        
        # 앙상블 가중치 (학습 가능)
        self.ensemble_weights = nn.Parameter(torch.ones(num_esp) / num_esp)
        
    def forward(self, x):
        # 입력 형태: (batch_size, num_esp, time_steps, subcarriers)
        batch_size = x.size(0)
        num_esp = x.size(1)
        
        # 각 ESP 모델의 예측 결과
        esp_outputs = []
        for i in range(num_esp):
            esp_data = x[:, i, :, :].unsqueeze(1)
            output = self.esp_models[i](esp_data)
            esp_outputs.append(output)
        
        # 앙상블 가중치 정규화
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # 가중 평균 계산
        ensemble_output = torch.zeros_like(esp_outputs[0])
        for i in range(num_esp):
            ensemble_output += weights[i] * esp_outputs[i]
        
        return ensemble_output


# 모델 테스트 함수
def test_cnn_models():
    # 테스트 데이터 생성
    batch_size = 8
    num_esp = 4
    time_steps = 180
    subcarriers = 114
    num_classes = 3
    
    # 랜덤 입력 데이터
    x = torch.randn(batch_size, num_esp, time_steps, subcarriers)
    
    # 기본 CNN 모델
    model1 = WiFiCSICNN(num_classes=num_classes)
    output1 = model1(x)
    print(f"기본 CNN 모델 출력 크기: {output1.shape}")
    
    # Attention CNN 모델
    model2 = WiFiCSICNNAttention(num_classes=num_classes)
    output2 = model2(x)
    print(f"Attention CNN 모델 출력 크기: {output2.shape}")
    
    # 앙상블 CNN 모델
    model3 = WiFiCSICNNEnsemble(num_classes=num_classes)
    output3 = model3(x)
    print(f"앙상블 CNN 모델 출력 크기: {output3.shape}")
    
    # 모델 파라미터 수 계산
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"기본 CNN 모델 파라미터 수: {count_parameters(model1):,}")
    print(f"Attention CNN 모델 파라미터 수: {count_parameters(model2):,}")
    print(f"앙상블 CNN 모델 파라미터 수: {count_parameters(model3):,}")


if __name__ == "__main__":
    test_cnn_models()
