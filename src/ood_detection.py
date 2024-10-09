import torch
from models import ResNet  # 학습에 사용했던 모델 클래스 임포트
import argparse

def load_model(model_path):
    # 모델 아키텍처 초기화 (예: ResNet)
    net = ResNet()

    # 저장된 모델 가중치 로드
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()  # 모델을 평가 모드로 전환

    return net

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
    args = parser.parse_args()

    # 모델 로드
    model = load_model(args.model_path)

    # 예측 코드 (테스트 데이터 예시)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 예제 입력 (배치 크기, 채널 수, 이미지 높이, 이미지 너비)
    example_input = torch.randn(1, 3, 32, 32).to(device)

    with torch.no_grad():
        output = model(example_input)
        _, predicted = torch.max(output, 1)
        print(f'Predicted class: {predicted.item()}')

if __name__ == "__main__":
    main()
