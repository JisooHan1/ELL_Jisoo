import os
import json

def preprocess_tiny_imagenet(root='./datasets/tiny-imagenet-200'):
    """Tiny ImageNet-200의 train/val 데이터셋을 JSON 파일로 저장"""
    dataset_info = {'train': [], 'val': []}

    # Train 데이터 처리
    train_root = os.path.join(root, 'train')
    cls_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_root)))}

    for cls, idx in cls_idx.items():
        cls_path = os.path.join(train_root, cls, 'images')
        for img in os.listdir(cls_path):
            dataset_info['train'].append((os.path.join(cls_path, img), idx))

    # Validation 데이터 처리
    val_root = os.path.join(root, 'val')
    val_images_path = os.path.join(val_root, 'images')
    val_labels_path = os.path.join(val_root, 'val_annotations.txt')

    with open(val_labels_path, 'r') as f:
        for line in f:
            image_name, label, *_ = line.strip().split('\t')
            image_path = os.path.join(val_images_path, image_name)
            dataset_info['val'].append((image_path, cls_idx[label]))

    # JSON 파일로 저장
    json_path = os.path.join(root, 'tiny_imagenet_preprocessed.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_info, f)

    print(f" 전처리 완료! JSON 파일 저장: {json_path}")

    # 처리 후 이미지 수 검증
    print(f"Train samples: {len(dataset_info['train'])}")
    print(f"Validation samples: {len(dataset_info['val'])}")

    # 실제 파일 존재 여부 검증 (예시)
    for path, _ in dataset_info['train'] + dataset_info['val']:
        if not os.path.exists(path):
            print(f"경고: 누락된 이미지 파일 - {path}")

preprocess_tiny_imagenet()
