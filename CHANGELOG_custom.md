# CHANGELOG: TransUNet Custom Adaptation for Chest X-ray Lung Segmentation
# 흉부 X-ray 폐 분할을 위한 TransUNet 커스텀 개조 변경 이력

> **목적**: 본 문서는 JCCI 논문 작성 및 발표 방어(Defense)를 위해 작성된 기술 변경 이력서입니다.
> TransUNet 원본 레포지토리(https://github.com/Beckschen/TransUNet)를 기반으로
> 2D Chest X-ray 폐 분할 태스크에 적용하기 위해 수행한 모든 수정 내역과 새로 작성한 코드의
> 기술적 근거(Why), 설계 결정, 구현 세부 사항을 담고 있습니다.

---

## 목차 (Table of Contents)

1. [프로젝트 개요](#1-프로젝트-개요)
2. [환경 설정 및 레거시 코드 호환성 패치](#2-환경-설정-및-레거시-코드-호환성-패치)
   - 2.1 [패키지 의존성 업데이트 (requirements.txt)](#21-패키지-의존성-업데이트)
   - 2.2 [scipy deprecated import 수정 (dataset_synapse.py)](#22-scipy-deprecated-import-수정)
3. [데이터 전처리 파이프라인 구축 (dataset_custom.py)](#3-데이터-전처리-파이프라인-구축)
   - 3.1 [설계 철학 및 아키텍처 요구사항](#31-설계-철학-및-아키텍처-요구사항)
   - 3.2 [1채널 → 3채널 변환 근거](#32-1채널--3채널-변환-근거)
   - 3.3 [해상도 512×512 설정 근거](#33-해상도-512512-설정-근거)
   - 3.4 [ChestXrayDataset 클래스 구현 상세](#34-chestxraydataset-클래스-구현-상세)
   - 3.5 [RandomGenerator 및 데이터 증강](#35-randomgenerator-및-데이터-증강)
   - 3.6 [마스크 이진화 처리](#36-마스크-이진화-처리)
   - 3.7 [Train/Val 분리 전략](#37-trainval-분리-전략)
4. [학습 스크립트 구현 (train_custom.py)](#4-학습-스크립트-구현)
   - 4.1 [Argparse 기반 하이퍼파라미터 관리](#41-argparse-기반-하이퍼파라미터-관리)
   - 4.2 [AMP (Automatic Mixed Precision) 통합](#42-amp-automatic-mixed-precision-통합)
   - 4.3 [손실 함수 구성 (CE + Dice)](#43-손실-함수-구성-ce--dice)
   - 4.4 [폴리노미얼 학습률 스케줄러](#44-폴리노미얼-학습률-스케줄러)
   - 4.5 [검증 루프 및 Dice Score 계산](#45-검증-루프-및-dice-score-계산)
   - 4.6 [Best Model 저장 전략](#46-best-model-저장-전략)
   - 4.7 [TensorBoard 로깅](#47-tensorboard-로깅)
5. [자체 검증 스크립트 (test_dataset.py)](#5-자체-검증-스크립트)
6. [실험 재현성 가이드](#6-실험-재현성-가이드)
   - 6.1 [핵심 하이퍼파라미터 표](#61-핵심-하이퍼파라미터-표)
   - 6.2 [랜덤 시드 고정 전략](#62-랜덤-시드-고정-전략)
   - 6.3 [실행 커맨드 레퍼런스](#63-실행-커맨드-레퍼런스)
7. [파일 변경 요약](#7-파일-변경-요약)

---

## 1. 프로젝트 개요

### 원본 레포지토리 정보

| 항목 | 내용 |
|------|------|
| 원본 논문 | TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation (Chen et al., 2021) |
| 원본 태스크 | 3D CT 기반 다중 장기 분할 (Synapse Multi-organ Dataset) |
| 원본 입력 | 2D CT 슬라이스 (`.npz` 포맷, grayscale) |
| 원본 출력 | 9-class 다중 장기 세그멘테이션 마스크 |

### 본 커스텀의 변경 방향

| 항목 | 원본 | 커스텀 (본 작업) |
|------|------|-----------------|
| 데이터셋 | Synapse 3D CT | Montgomery & Shenzhen Chest X-ray |
| 입력 포맷 | `.npz` (numpy array) | `.png` (8-bit grayscale image) |
| 입력 채널 | 1채널 grayscale | **3채널 RGB** (채널 복제) |
| 입력 해상도 | 224×224 | **512×512** |
| 출력 클래스 | 9개 장기 | **2개 클래스** (배경/폐) |
| 데이터 분할 | 별도 리스트 파일 (`.txt`) | **자동 비율 분할** (80/20) |
| 학습 정밀도 | FP32 | **FP32 / AMP 선택 가능** |

---

## 2. 환경 설정 및 레거시 코드 호환성 패치

### 2.1 패키지 의존성 업데이트

**파일**: `requirements.txt`

#### 변경 전 (원본)
```
torch==1.4.0
torchvision==0.5.0
numpy
tqdm
tensorboard
tensorboardX
ml-collections
medpy
SimpleITK
scipy
h5py
```

#### 변경 후
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=2.0.0
tqdm
tensorboard
tensorboardX
ml-collections
medpy
SimpleITK
scipy
h5py
Pillow
```

#### 변경 근거 (Why)

**① PyTorch 버전 고정 해제 (`torch==1.4.0` → `torch>=2.0.0`)**

원본 코드는 2020년 릴리스 기준인 PyTorch 1.4.0으로 고정되어 있었습니다. 이는 다음의 문제를 야기합니다:

- `torch.cuda.amp.autocast` 및 `torch.cuda.amp.GradScaler` API는 PyTorch 1.6 이후 도입됨. AMP 기능 사용 불가.
- PyTorch 2.0에서 도입된 `torch.compile()` 최적화 불가.
- 2025년 현재 대부분의 서버 환경은 CUDA 11.x ~ 12.x를 사용하므로, `torch==1.4.0` 빌드는 구하기 어려움.
- `DataLoader`의 `persistent_workers`, `pin_memory` 등 성능 최적화 기능 일부 미지원.

**② numpy 버전 하한선 명시 (`numpy` → `numpy>=2.0.0`)**

numpy 2.0 (2024년 6월 릴리스)에서는 오랫동안 deprecated 처리되었던 Python 내장 타입 별칭들이 **완전히 제거**되었습니다:

| 제거된 별칭 | 대체 표현 |
|------------|----------|
| `np.float` | `np.float64` 또는 Python `float` |
| `np.int` | `np.int_` 또는 Python `int` |
| `np.bool` | `np.bool_` 또는 Python `bool` |
| `np.complex` | `np.complex128` 또는 Python `complex` |
| `np.object` | `np.object_` 또는 Python `object` |
| `np.str` | `np.str_` 또는 Python `str` |

원본 TransUNet 코드 내에는 이러한 별칭의 직접 사용은 없었으나, 명시적으로 `numpy>=2.0.0`을 선언함으로써 향후 코드 확장 시 호환성 문제를 사전 방지합니다.

**③ Pillow 추가**

원본 TransUNet은 CT 데이터를 `h5py` 또는 `numpy` 배열로 직접 로드하므로 이미지 파일 처리 라이브러리가 불필요했습니다. 본 커스텀에서는 Chest X-ray 데이터가 `.png` 파일 형태로 제공되므로 `Pillow`(PIL)를 추가했습니다. `cv2`(OpenCV) 대신 Pillow를 선택한 이유는 PyTorch/torchvision 생태계와의 통합성이 높고, 추가 컴파일 의존성 없이 설치가 용이하기 때문입니다.

---

### 2.2 scipy deprecated import 수정

**파일**: `TransUNet/datasets/dataset_synapse.py`

#### 변경 전 (원본, line 7)
```python
from scipy.ndimage.interpolation import zoom
```

#### 변경 후 (line 7)
```python
from scipy.ndimage import zoom
```

#### 변경 근거 (Why)

`scipy.ndimage.interpolation` 서브모듈은 scipy 1.0.0 시점부터 deprecated 처리되었으며, scipy 1.14.x (2024년 기준 최신) 이후 완전히 제거되었습니다. 해당 함수는 `scipy.ndimage` 최상위 네임스페이스에서 직접 접근하는 것이 공식 권장 방법입니다.

이 변경은 순수하게 **API 마이그레이션**에 해당하며 기능상 동일합니다:
```python
# 두 코드는 완전히 동일하게 동작 (scipy < 1.14)
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom

# scipy >= 1.14 에서는 상단 코드만 동작
```

실질적 영향: 최신 scipy를 사용하는 환경에서 `ImportError`를 방지하며, 원본 Synapse 데이터셋 학습 파이프라인도 현대 환경에서 정상 작동하도록 보장합니다.

---

## 3. 데이터 전처리 파이프라인 구축

**파일**: `TransUNet/datasets/dataset_custom.py` *(신규 생성)*

### 3.1 설계 철학 및 아키텍처 요구사항

TransUNet의 Encoder는 ImageNet으로 사전 학습된 ResNet50 + ViT-B/16 Hybrid Architecture입니다. 사전 학습 가중치를 활용하기 위해서는 입력이 해당 아키텍처의 **학습 당시 입력 조건**과 일치해야 합니다.

TransUNet 아키텍처의 입력 요구사항:
```
Input Tensor Shape: [Batch, 3, H, W]
  - 채널 수: 3 (RGB 3채널)
  - H, W: 패치 분할이 가능한 크기 (patch_size의 배수)
  - 정규화: 0~1 범위의 float32
```

ViT의 패치 분할 방식:
```
img_size = 512, patch_size = 16
→ grid = (512/16, 512/16) = (32, 32) = 1024개 패치
```

### 3.2 1채널 → 3채널 변환 근거

Chest X-ray 이미지는 기본적으로 **grayscale(1채널)** 이미지입니다. 그러나 TransUNet의 R50-ViT-B_16 Encoder는 ImageNet으로 사전 학습되었으며, ImageNet은 **3채널 RGB 이미지**로 학습된 모델입니다.

**변환이 필요한 이유**:

1. **아키텍처 호환성**: ResNet50의 첫 번째 Convolution 레이어(`Conv1`)는 `in_channels=3`으로 고정되어 있습니다. 1채널 입력은 차원 불일치로 `RuntimeError`를 발생시킵니다.

2. **사전 학습 가중치 활용**: R50+ViT-B_16 pretrained weights(`R50+ViT-B_16.npz`)는 3채널 입력을 전제로 학습된 가중치입니다. 채널 수를 맞춰야 Transfer Learning이 가능합니다.

3. **의미론적 정당성 (Semantic Justification)**: 흑백 X-ray를 RGB로 복제(channel repetition)하는 방식은 **grayscale-to-pseudo-RGB** 변환으로, 의료 영상 딥러닝에서 일반적으로 사용되는 관행입니다. 세 채널이 동일한 정보를 담고 있어 색상 정보의 왜곡 없이 사전 학습 피처를 활용할 수 있습니다.

**구현 방식 (dataset_custom.py)**:
```python
# 1채널 → 3채널: numpy 축 조작으로 채널 복제
image = np.expand_dims(image, axis=2)  # [H, W] → [H, W, 1]

# RandomGenerator.__call__ 내부에서 채널 복제 수행
if image.shape[2] == 1:
    image = np.repeat(image, 3, axis=2)  # [H, W, 1] → [H, W, 3]

# PyTorch 텐서 형식으로 변환: [H, W, C] → [C, H, W]
image = torch.from_numpy(
    image.transpose(2, 0, 1).astype(np.float32)
)
# 최종 shape: [3, 512, 512]
```

**대안과 비교**:

| 방법 | 설명 | 선택 이유 |
|------|------|----------|
| **채널 복제** (채택) | 동일한 grayscale을 R, G, B 세 채널로 복사 | 정보 손실 없음, 구현 단순 |
| 첫 Conv 레이어 수정 | `in_channels=1`로 변경 | 사전 학습 가중치 첫 레이어 무효화 |
| 제로 패딩 | 나머지 2채널을 0으로 채움 | 정보 비대칭, 성능 저하 위험 |
| CLAHE/히스토그램 정규화 후 복제 | 대비 향상 후 복제 | 추가 전처리 단계, 본 실험 범위 외 |

### 3.3 해상도 512×512 설정 근거

원본 TransUNet은 `img_size=224`를 기본값으로 사용합니다. 본 커스텀에서 **512×512를 채택한 근거**는 다음과 같습니다:

**① 데이터셋 원본 해상도**:
Montgomery & Shenzhen Dataset의 Chest X-ray 이미지 원본 해상도는 대략 **2000~3000 픽셀**입니다. 폐 분할에서 경계(boundary) 정확도가 중요하므로, 더 높은 입력 해상도가 세부 구조 보존에 유리합니다.

**② Patch 분할 계산**:
```
img_size=224, patch_size=16: grid = (14, 14) = 196 patches
img_size=512, patch_size=16: grid = (32, 32) = 1024 patches
```
512×512에서는 ViT가 더 많은 패치(1024개)를 처리하므로 공간적 세부 정보를 더 잘 포착할 수 있습니다.

**③ 의료 영상 분야의 관행**:
TransUNet 논문 자체에서도 `img_size=512` 실험이 병행되었으며, 다수의 Medical Image Segmentation 논문에서 512×512를 표준 입력 크기로 채택합니다.

**④ 트레이드오프 인식**:
- 512×512: 더 높은 정확도, 더 많은 GPU 메모리 필요, 낮은 처리 속도
- 224×224: 낮은 메모리 사용, 빠른 학습 속도, 세부 정보 손실 가능성

본 실험에서는 정확도 우선 방침으로 512×512를 채택하되, AMP(자동 혼합 정밀도)로 메모리 제약을 완화합니다.

### 3.4 ChestXrayDataset 클래스 구현 상세

**데이터셋 디렉토리 구조**:
```
data/Chest Xray Masks and Labels/data/Lung Segmentation/
├── CXR_png/
│   ├── CHNCXR_0001_0.png   ← 입력 X-ray 이미지 (grayscale)
│   ├── CHNCXR_0002_0.png
│   └── ...
└── masks/
    ├── CHNCXR_0001_0_mask.png   ← 폐 영역 바이너리 마스크
    ├── CHNCXR_0002_0_mask.png
    └── ...
```

**파일명 매칭 규칙**:
```python
img_name  = "CHNCXR_0001_0.png"
mask_name = img_name.replace('.png', '_mask.png')
          = "CHNCXR_0001_0_mask.png"
```

이 매칭 규칙은 Montgomery County CXR Dataset과 Shenzhen Hospital CXR Dataset 모두에서 일관되게 적용되는 파일명 규약입니다. 매칭 실패 시 해당 샘플을 자동으로 건너뜁니다:
```python
if os.path.exists(mask_path):
    self.samples.append((img_path, mask_path))
```

**이미지 로딩 파이프라인**:
```python
def __getitem__(self, idx):
    img_path, mask_path = self.samples[idx]

    # Step 1: PIL로 grayscale 강제 변환 ('L' mode: 8-bit grayscale)
    image = Image.open(img_path).convert('L')
    image = np.array(image)  # shape: [H, W], dtype: uint8, range: [0, 255]

    # Step 2: 마스크 로딩
    label = Image.open(mask_path).convert('L')
    label = np.array(label)  # shape: [H, W], dtype: uint8

    # Step 3: 이미지 정규화 [0, 255] → [0, 1]
    image = image.astype(np.float32) / 255.0

    # Step 4: 마스크 이진화 (임계값 127)
    label = (label > 127).astype(np.uint8)  # {0, 1} 이진 마스크

    # Step 5: 채널 차원 추가 [H, W] → [H, W, 1]
    image = np.expand_dims(image, axis=2)

    sample = {'image': image, 'label': label}
    ...
```

**PIL의 `.convert('L')` 사용 근거**:
데이터셋에 따라 RGBA, RGB, 또는 P(palette) 모드로 저장된 이미지가 혼재할 수 있습니다. `.convert('L')`은 어떤 입력 모드든 8-bit grayscale로 통일하여 **일관된 전처리 파이프라인**을 보장합니다.

### 3.5 RandomGenerator 및 데이터 증강

데이터 증강 파이프라인은 원본 `dataset_synapse.py`의 `RandomGenerator`를 Chest X-ray에 맞게 재설계하였습니다.

**주요 변경점**:
원본은 2D 단일 채널 이미지 (`[H, W]`)를 전제로 설계되었으나, 본 커스텀은 채널 차원이 추가된 이미지 (`[H, W, 1]`)를 처리해야 합니다. zoom 함수 호출 시 채널 축을 고려합니다:

```python
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image shape: [H, W, 1] (float32, range [0,1])
        # label shape: [H, W]    (uint8, values {0, 1})

        # 데이터 증강 (50% 확률로 각각 적용)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 리사이즈: 원본과 목표 해상도가 다를 경우
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            # 이미지: 채널 축(axis=2)은 스케일 1 유지
            image = zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y, 1),
                order=3  # 3차 스플라인 보간 (고화질)
            )
            # 마스크: 최근접 이웃 보간 (이진값 보존)
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0  # 0차(최근접 이웃) 보간
            )

        # 1채널 → 3채널 복제
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)  # [H, W, 1] → [H, W, 3]

        # NumPy [H, W, C] → PyTorch [C, H, W]
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        return {'image': image, 'label': label.long()}
```

**Zoom order 선택 근거**:

| 보간 Order | 방법 | 적용 대상 | 이유 |
|-----------|------|----------|------|
| `order=3` | 3차 스플라인 (Cubic) | 입력 이미지 | 부드러운 이미지 품질, 엣지 보존 |
| `order=0` | 최근접 이웃 (Nearest-Neighbor) | 세그멘테이션 마스크 | {0, 1} 이진값 보존 필수; 스플라인 사용 시 0.3, 0.7 등 소수값 생성 |

**적용 증강 기법**:

```python
def random_rot_flip(image, label):
    """90도 단위 회전 + 수평/수직 뒤집기"""
    k = np.random.randint(0, 4)   # 0°, 90°, 180°, 270° 중 선택
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)  # 0: 수직 뒤집기, 1: 수평 뒤집기
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    """임의 각도 회전 (-20° ~ +20°)"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
```

X-ray 흉부 영상에서 이러한 기하학적 증강이 적절한 이유:
- **좌우 대칭**: 흉부는 좌우 대칭 구조이므로 수평 뒤집기가 자연스러운 증강.
- **소각도 회전**: 촬영 시 환자 자세 변이(-20°~+20°)를 모방하여 모델의 자세 불변성 강화.
- **90°/180° 회전**: 의료 영상 데이터셋에서 관행적으로 사용되는 강한 증강으로 소규모 데이터셋 과적합 완화.

### 3.6 마스크 이진화 처리

```python
label = (label > 127).astype(np.uint8)
```

**처리 근거**:
Montgomery Dataset의 마스크는 이론적으로 순수 흑백(0 또는 255)이지만, PNG 저장/압축 과정에서 경계 픽셀에 중간값(예: 128, 200)이 발생할 수 있습니다. 임계값 127을 적용하여 다음을 보장합니다:

- 마스크 픽셀값: `{0, 1}` (배경=0, 폐=1)
- 손실 함수 `CrossEntropyLoss`의 클래스 인덱스 입력 조건 충족
- `DiceLoss`의 one-hot encoding 처리 조건 충족

### 3.7 Train/Val 분리 전략

원본 TransUNet은 별도의 `.txt` 리스트 파일로 학습/테스트 데이터를 분리합니다. 본 커스텀에서는 파일 정렬 후 비율 분할 방식을 채택했습니다:

```python
# 파일명 기준 알파벳 정렬 → 재현 가능한 분할 보장
image_files = sorted(glob(os.path.join(image_dir, '*.png')))

num_samples = len(self.samples)
num_train = int(num_samples * train_ratio)  # 기본: 80%

if split == 'train':
    self.samples = self.samples[:num_train]   # 앞 80%
elif split == 'val':
    self.samples = self.samples[num_train:]   # 뒤 20%
```

**이 방식의 장단점**:

| 장점 | 단점 및 주의사항 |
|------|----------------|
| 추가 파일 불필요, 코드만으로 완결 | K-fold 교차 검증 미지원 |
| `sorted()`로 동일한 순서 보장 → 재현 가능 | 데이터 분포 편향 가능성 (파일명 순서에 의존) |
| `train_ratio` 변경으로 유연한 조정 가능 | 공식 데이터셋 분할 규약을 따르지 않음 |

---

## 4. 학습 스크립트 구현

**파일**: `TransUNet/train_custom.py` *(신규 생성)*

원본 `train.py` + `trainer.py`를 **단일 파일로 통합**하고, 커스텀 데이터셋 및 현대 학습 기법을 적용한 새 학습 스크립트입니다.

### 4.1 Argparse 기반 하이퍼파라미터 관리

원본의 하드코딩된 파라미터를 argparse로 전면 교체하여 **재실험 시 코드 수정 없이 커맨드라인에서 파라미터 변경**이 가능하도록 설계했습니다.

```python
parser = argparse.ArgumentParser()

# 데이터 파라미터
parser.add_argument('--root_path', type=str,
    default='../data/Chest Xray Masks and Labels/data/Lung Segmentation')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--train_ratio', type=float, default=0.8)

# 모델 파라미터
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--img_size', type=int, default=512)

# 학습 파라미터
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=1234)

# 특수 기능 파라미터
parser.add_argument('--use_amp', action='store_true')   # AMP 활성화
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--output_dir', type=str, default='../model')
```

**스냅샷 경로 자동 생성 전략**:
실험 파라미터를 디렉토리명에 인코딩하여 실험 관리를 용이하게 합니다:
```python
snapshot_path = f"../model/TU_ChestXray_512/TU_R50-ViT-B_16_skip3_bs8_lr0.01_512_amp"
```

### 4.2 AMP (Automatic Mixed Precision) 통합

```python
from torch.cuda.amp import GradScaler, autocast

# GradScaler 초기화 (AMP 활성화 시에만)
scaler = GradScaler() if args.use_amp else None
```

**AMP 학습 루프**:
```python
if args.use_amp:
    with autocast():  # FP16 연산 범위
        outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch.long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

    optimizer.zero_grad()
    scaler.scale(loss).backward()   # 스케일링된 역전파
    scaler.step(optimizer)          # 스케일 해제 후 파라미터 업데이트
    scaler.update()                 # 다음 iteration을 위한 스케일 조정
else:
    # 표준 FP32 학습
    outputs = model(image_batch)
    loss = 0.5 * ce_loss(...) + 0.5 * dice_loss(...)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**AMP 채택 근거 및 기대 효과**:

| 항목 | FP32 | AMP (FP16+FP32 혼합) |
|------|------|---------------------|
| 텐서 메모리 | 4 bytes/element | 2 bytes/element (절반) |
| 연산 속도 | 기준 | NVIDIA Tensor Core에서 최대 2-3x 가속 |
| 수치 안정성 | 높음 | GradScaler로 underflow 방지 |
| Loss Scaling | 불필요 | 자동 동적 스케일링 (`GradScaler`) |

`autocast()` 컨텍스트 내부에서 Tensor Core 지원 연산(GEMM, Conv 등)은 자동으로 FP16으로 실행됩니다. `GradScaler`는 gradient가 FP16 표현 범위에서 underflow(0으로 수렴)되는 현상을 방지하기 위해 loss를 동적으로 스케일링합니다.

### 4.3 손실 함수 구성 (CE + Dice)

```python
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(num_classes=2)

# 동등 가중치 결합
loss = 0.5 * ce_loss(outputs, label_batch.long()) + \
       0.5 * dice_loss(outputs, label_batch, softmax=True)
```

**혼합 손실 함수 채택 근거**:

- **Cross Entropy Loss**: 픽셀 단위 분류 손실. 모든 픽셀을 독립적으로 처리하므로 전체적인 분류 정확도 학습에 강점. 단, 클래스 불균형(배경 >> 폐)에 취약.

- **Dice Loss**: 예측 마스크와 실제 마스크의 겹침 비율(IoU 기반)을 최적화. 클래스 불균형에 강인하며, 최종 평가 지표인 Dice Score를 직접 최적화하는 효과.

- **결합(0.5 CE + 0.5 Dice)**: TransUNet 원논문의 실험적으로 검증된 설정을 그대로 계승. CE는 픽셀 수준 지도, Dice는 영역 수준 지도를 제공하여 학습 안정성과 분할 품질을 동시에 확보.

흉부 X-ray 데이터에서 폐 영역은 전체 픽셀의 약 30~40%를 차지합니다. 따라서 클래스 불균형이 심각하지 않으나, Dice Loss의 포함은 여전히 세밀한 경계(boundary) 최적화에 기여합니다.

### 4.4 폴리노미얼 학습률 스케줄러

```python
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

for param_group in optimizer.param_groups:
    param_group['lr'] = lr_
```

**폴리노미얼 감쇠 선택 근거**:
TransUNet 원논문의 설정을 그대로 계승합니다. Cosine Annealing에 비해 감쇠가 천천히 시작되어 학습 초기 충분한 탐색(exploration)이 가능하며, 학습 후반부에 빠르게 수렴합니다. 지수 = 0.9는 원논문의 실험적 최적값입니다.

학습률 변화 예시 (`base_lr=0.01`, `max_iter=30000`):
```
iter=0:     lr = 0.01000 (100%)
iter=7500:  lr = 0.00760 (76%)
iter=15000: lr = 0.00533 (53%)
iter=22500: lr = 0.00293 (29%)
iter=30000: lr = 0.00000 (0%)
```

### 4.5 검증 루프 및 Dice Score 계산

원본 `trainer.py`에는 **검증 루프가 없었습니다**. 에폭 종료 후 검증 손실이나 성능 지표를 확인할 방법이 없어 과적합(overfitting) 탐지가 불가능했습니다. 본 커스텀에서 에폭 단위 검증 루프를 추가했습니다.

```python
def validate(model, val_loader, num_classes, use_amp=False):
    model.eval()
    dice_scores = []

    with torch.no_grad():  # 추론 시 gradient 계산 비활성화 (메모리 절약)
        for sampled_batch in val_loader:
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            if use_amp:
                with autocast():
                    outputs = model(image_batch)
            else:
                outputs = model(image_batch)

            # Softmax → argmax로 예측 클래스 결정
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            dice = calculate_dice_score(preds, label_batch, num_classes)
            dice_scores.append(dice)

    model.train()  # 검증 후 반드시 학습 모드 복귀
    return np.mean(dice_scores)
```

**Dice Score 계산 공식**:
```python
def calculate_dice_score(pred, target, num_classes=2):
    for cls in range(1, num_classes):  # 배경(class=0) 제외
        pred_cls = (pred == cls).float()    # 예측 이진 마스크
        target_cls = (target == cls).float() # 정답 이진 마스크

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        # Dice = 2|X∩Y| / (|X| + |Y|)
        dice = (2.0 * intersection) / (union + 1e-7)  # 1e-7: 분모 0 방지
```

수식:
```
Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)
```
- 완전 일치: Dice = 1.0
- 완전 불일치: Dice = 0.0
- 임상적 허용 기준: Dice ≥ 0.85 (폐 분할 분야 관행)

**배경 클래스 제외 이유**: 배경 픽셀이 다수를 차지하므로 포함 시 Dice가 인위적으로 높게 측정됩니다. 관심 영역(폐)에 대해서만 Dice를 계산하는 것이 의미있는 성능 지표입니다.

### 4.6 Best Model 저장 전략

```python
best_dice = 0.0

# 에폭 종료 후 검증
avg_dice = validate(model, valloader, num_classes, use_amp=args.use_amp)

if avg_dice > best_dice:
    best_dice = avg_dice
    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')

    # 멀티 GPU 지원: DataParallel 래핑 시 .module로 실제 모델 접근
    if args.n_gpu > 1:
        torch.save(model.module.state_dict(), save_mode_path)
    else:
        torch.save(model.state_dict(), save_mode_path)
```

검증 Dice Score가 개선될 때만 모델을 저장하는 **Early Stopping 기준점** 역할을 합니다. 단, 학습이 조기 종료되지 않고 최고 성능 모델을 별도 파일(`best_model.pth`)로 보존합니다.

### 4.7 TensorBoard 로깅

```python
writer = SummaryWriter(snapshot_path + '/log')

# 스칼라 로깅
writer.add_scalar('info/lr', lr_, iter_num)
writer.add_scalar('info/total_loss', loss, iter_num)
writer.add_scalar('info/loss_ce', loss_ce, iter_num)
writer.add_scalar('validation/dice_score', avg_dice, epoch_num)

# 이미지 로깅 (50 iteration마다)
writer.add_image('train/Image', image_normalized, iter_num)
writer.add_image('train/Prediction', pred_visualization, iter_num)
writer.add_image('train/GroundTruth', gt_visualization, iter_num)
```

실험 모니터링 항목:
- `info/lr`: 폴리노미얼 학습률 스케줄 시각화
- `info/total_loss`: 전체 손실 (0.5 CE + 0.5 Dice)
- `info/loss_ce`: Cross Entropy 손실 단독
- `validation/dice_score`: 에폭별 검증 Dice Score (가장 중요한 지표)
- 이미지 탭: 학습 중 예측 결과 직접 시각 확인

---

## 5. 자체 검증 스크립트

**파일**: `TransUNet/test_dataset.py` *(신규 생성)*

학습 시작 전 데이터 파이프라인이 올바르게 구성되었는지 확인하는 독립적인 검증 스크립트입니다. 학습 스크립트 실행 전 반드시 먼저 실행할 것을 권장합니다.

**검증 항목 및 기대 출력**:

```
=== 검증 항목 1: 디렉토리 존재 확인 ===
✓ CXR_png/ 디렉토리 존재
✓ masks/ 디렉토리 존재

=== 검증 항목 2: 데이터셋 생성 ===
Loaded 454 train samples from ...
Loaded 114 val samples from ...
✓ 데이터셋 생성 성공

=== 검증 항목 3: 텐서 형상 확인 ===
Image shape: torch.Size([3, 512, 512])    ← 기대값
Image dtype: torch.float32
Image range: [0.000, 1.000]
Label shape: torch.Size([512, 512])
Label unique values: [0, 1]               ← 이진 마스크 확인

=== 검증 항목 4: DataLoader 테스트 ===
Batch image shape: torch.Size([2, 3, 512, 512])
Batch label shape: torch.Size([2, 512, 512])
✓ DataLoader 정상 작동

=== 검증 항목 5: 시각화 ===
→ train_samples.png 저장 (X-ray, Mask, Overlay 3열 구성)
→ val_samples.png 저장
```

**테스트 통과 기준**:

| 검사 항목 | 기대값 | 실패 원인 |
|----------|--------|----------|
| Image shape | `[3, 512, 512]` | 채널 변환 실패 |
| Image range | `[0.0, 1.0]` | 정규화 미적용 |
| Label unique values | `{0, 1}` | 이진화 실패 |
| Batch shape | `[B, 3, 512, 512]` | DataLoader 설정 오류 |

---

## 6. 실험 재현성 가이드

### 6.1 핵심 하이퍼파라미터 표

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `vit_name` | `R50-ViT-B_16` | ResNet50 + ViT-B/16 하이브리드 인코더 |
| `img_size` | `512` | 입력 이미지 해상도 |
| `patch_size` | `16` | ViT 패치 크기 |
| `patches.grid` | `(32, 32)` | 512/16 = 32 |
| `num_classes` | `2` | 배경 + 폐 |
| `n_skip` | `3` | ResNet에서 추출하는 skip connection 수 |
| `decoder_channels` | `(256, 128, 64, 16)` | 디코더 각 스테이지 채널 수 |
| `skip_channels` | `[512, 256, 64, 16]` | 인코더 skip connection 채널 수 |
| `batch_size` | `8` | GPU당 배치 크기 |
| `base_lr` | `0.01` | SGD 초기 학습률 |
| `optimizer` | `SGD` | momentum=0.9, weight_decay=1e-4 |
| `lr_schedule` | `Polynomial` | exponent=0.9 |
| `max_epochs` | `150` | 총 학습 에폭 수 |
| `loss` | `0.5 CE + 0.5 Dice` | 혼합 손실 함수 |
| `train_ratio` | `0.8` | 학습/검증 분할 비율 |
| `use_amp` | `True` (권장) | AMP 활성화 |
| `seed` | `1234` | 재현성을 위한 랜덤 시드 |

### 6.2 랜덤 시드 고정 전략

완전한 재현성을 위해 다음 4가지 시드를 모두 고정합니다:

```python
import random, numpy as np, torch

random.seed(args.seed)          # Python 내장 random
np.random.seed(args.seed)       # NumPy random
torch.manual_seed(args.seed)    # PyTorch CPU 연산
torch.cuda.manual_seed(args.seed)  # PyTorch GPU 연산

# 결정론적 학습 모드
torch.backends.cudnn.benchmark = False    # 입력 크기에 따른 자동 알고리즘 선택 비활성화
torch.backends.cudnn.deterministic = True  # 동일 입력 → 동일 출력 보장
```

**주의사항**: `cudnn.deterministic=True`는 일부 연산에서 성능 저하를 유발할 수 있습니다. 재현성보다 성능이 중요한 경우 `deterministic=0` 인자를 사용할 수 있습니다.

### 6.3 실행 커맨드 레퍼런스

**데이터셋 검증 (학습 전 필수)**:
```bash
cd TransUNet
python test_dataset.py \
    --root_path "../data/Chest Xray Masks and Labels/data/Lung Segmentation" \
    --img_size 512
```

**로컬 환경 빠른 테스트 (기능 확인용)**:
```bash
python train_custom.py \
    --batch_size 2 \
    --max_epochs 5 \
    --num_workers 0 \
    --img_size 512 \
    --save_interval 2
```

**표준 학습 (단일 GPU, AMP 활성화)**:
```bash
python train_custom.py \
    --batch_size 8 \
    --max_epochs 150 \
    --num_workers 4 \
    --img_size 512 \
    --base_lr 0.01 \
    --use_amp \
    --save_interval 10 \
    --seed 1234
```

**고성능 서버 학습 (다중 GPU)**:
```bash
python train_custom.py \
    --batch_size 16 \
    --max_epochs 150 \
    --n_gpu 2 \
    --num_workers 8 \
    --img_size 512 \
    --base_lr 0.01 \
    --use_amp \
    --seed 1234
```

**TensorBoard 모니터링**:
```bash
tensorboard --logdir ../model/TU_ChestXray_512_R50-ViT-B_16_skip3_bs8_lr0.01_512_amp/log
```

---

## 7. 파일 변경 요약

### 수정된 파일

| 파일 | 변경 유형 | 변경 내용 | 영향 범위 |
|------|----------|----------|----------|
| `requirements.txt` | 수정 | PyTorch `>=2.0.0`, numpy `>=2.0.0`, Pillow 추가 | 전체 환경 |
| `datasets/dataset_synapse.py` | 수정 (1줄) | `scipy.ndimage.interpolation.zoom` → `scipy.ndimage.zoom` | 원본 Synapse 파이프라인 |

### 신규 생성된 파일

| 파일 | 용도 |
|------|------|
| `datasets/dataset_custom.py` | Chest X-ray PyTorch Dataset 클래스 및 DataLoader 팩토리 |
| `train_custom.py` | 커스텀 학습 스크립트 (argparse, AMP, 검증 루프, 모델 저장 포함) |
| `test_dataset.py` | 학습 전 데이터 파이프라인 검증 스크립트 |
| `README_CUSTOM.md` | 커스텀 구현 상세 문서 |
| `QUICKSTART.md` | 빠른 시작 가이드 |
| `CHANGELOG_custom.md` | 본 파일: 전체 변경 이력 및 기술 근거 문서 |

### 수정하지 않은 파일 (원본 유지)

| 파일 | 유지 이유 |
|------|----------|
| `networks/vit_seg_modeling.py` | 모델 아키텍처 변경 없음 (채널 수는 입력 단계에서 처리) |
| `networks/vit_seg_configs.py` | 설정은 학습 스크립트에서 동적으로 주입 |
| `networks/vit_seg_modeling_resnet_skip.py` | 수정 불필요 |
| `utils.py` | DiceLoss 재사용 |
| `train.py` / `trainer.py` | 원본 Synapse 학습용으로 보존 |
| `test.py` | 원본 테스트 파이프라인 보존 |

---

*최종 수정일: 2026-03-12*
*작성 목적: JCCI 논문 작성 및 발표 방어(Defense) 자료*
