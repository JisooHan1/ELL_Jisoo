import torch
import torch.nn.functional as F
from datasets import load_dataset  # datasets.py에서 load_dataset 함수 임포트
from models import ResNet  # 학습에 사용했던 모델 클래스 임포트
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def load_model(model_path):
    # 모델 아키텍처 초기화 (예: ResNet)
    net = ResNet(3)

    # 저장된 모델 가중치 로드
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()  # 모델을 평가 모드로 전환

    return net

def is_ood(input_data, model):
    # 모델을 평가 모드로 설정
    model.eval()

    # 예측 수행
    with torch.no_grad():
        outputs = model(input_data)
        softmax_scores = F.softmax(outputs, dim=1)  # Softmax 스코어 계산
        max_score, _ = torch.max(softmax_scores, dim=1)  # 최대 Softmax 스코어 추출

    return max_score.item()

def evaluate_ood_detection(id_scores, ood_scores):
    # 정답 라벨 생성: ID 데이터는 0, OOD 데이터는 1로 라벨링
    labels = [0] * len(id_scores) + [1] * len(ood_scores)
    scores = id_scores + ood_scores

    # AUROC 계산
    auroc = roc_auc_score(labels, scores)

    # AUPR 계산
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)

    print(f'AUROC: {auroc:.4f}')
    print(f'AUPR: {aupr:.4f}')

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model with OOD detection")
    parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("-id", "--id_dataset", type=str, required=True, help="Name of the in-distribution dataset (e.g., CIFAR10)")
    parser.add_argument("-ood", "--ood_dataset", type=str, required=True, help="Name of the out-of-distribution dataset (e.g., SVHN)")
    args = parser.parse_args()

    # 모델 로드
    model = load_model(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ID와 OOD 데이터셋 로드 (테스트셋만 사용)
    _, id_testset = load_dataset(args.id_dataset)
    _, ood_testset = load_dataset(args.ood_dataset)

    # DataLoader 생성 (테스트셋 사용)
    id_loader = torch.utils.data.DataLoader(id_testset, batch_size=1, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=1, shuffle=True)

    # ID와 OOD 데이터에 대한 Softmax 스코어 수집
    id_scores = [is_ood(data[0].to(device), model) for data in id_loader]
    ood_scores = [is_ood(data[0].to(device), model) for data in ood_loader]

    # AUROC와 AUPR 계산 및 출력
    evaluate_ood_detection(id_scores, ood_scores)

if __name__ == "__main__":
    main()
