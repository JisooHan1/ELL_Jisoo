from ood_methods.mds import MDS
import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 64, 3)  # 간단한 특징 추출

    def forward(self, x):
        return self.features(x)

# 테스트 코드
def test_mds():
    # 1. 더미 데이터 생성
    batch_size = 4
    num_samples = 10  # 클래스당 샘플 수

    # ID 학습용 데이터 생성 (CIFAR-10 형식: 3x32x32)
    dummy_inputs = torch.randn(batch_size, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (batch_size,))

    # 데이터로더 시뮬레이션
    class DummyDataLoader:
        def __init__(self, inputs, labels, num_samples):
            self.inputs = inputs
            self.labels = labels
            self.num_samples = num_samples

        def __iter__(self):
            for _ in range(self.num_samples):
                yield self.inputs, self.labels

    dummy_loader = DummyDataLoader(dummy_inputs, dummy_labels, num_samples)

    # 2. MDS 초기화 및 테스트
    model = DummyModel()
    mds = MDS(model)

    # 3. 메소드 테스트
    print("=== Testing MDS ===")

    # apply_method 테스트
    print("\nTesting apply_method...")
    mds.apply_method(dummy_loader)

    # OOD 점수 계산 테스트
    print("\nTesting OOD scoring...")
    test_input = torch.randn(2, 3, 32, 32)  # 테스트용 입력
    scores = mds.ood_score(test_input)
    print("OOD scores shape:", scores.shape)
    print("OOD scores:", scores)

if __name__ == "__main__":
    test_mds()