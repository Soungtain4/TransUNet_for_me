# 설치 가이드 — 패키지 목록 및 설치 명령어

> **환경 정보 (확인된 값)**
> - OS: Windows 11 Education
> - GPU: NVIDIA RTX 3060 Ti (VRAM 8GB)
> - CUDA Driver Version: 12.6
> - Python: 3.12.2

---

## 이미 설치되어 있는 패키지 ✅

별도 설치 불필요:

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `numpy` | 1.26.4 | 배열 연산 |
| `Pillow` | 10.2.0 | PNG 이미지 로딩 |
| `scipy` | 1.12.0 | 이미지 리사이즈 (zoom) |
| `h5py` | 3.10.0 | HDF5 파일 처리 (원본 코드) |
| `tensorboard` | 2.16.2 | 학습 시각화 |
| `tqdm` | 4.66.2 | 진행 바 |
| `jupyter` | 4.1.2 | 노트북 실행 환경 |
| `matplotlib` | (설치됨) | 시각화 |

---

## 설치가 필요한 패키지 ⚠️

### Step 1: PyTorch (CUDA 12.6 환경)

```bash
# 기존 손상 설치가 있다면 먼저 제거
pip uninstall torch torchvision torchaudio -y

# CUDA 12.4 빌드 설치 (CUDA 12.6 호환 가능한 최신 공식 빌드)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **왜 `cu124`인가?**
> PyTorch 공식 빌드는 CUDA 12.1, 12.4를 지원합니다.
> CUDA Driver 12.6은 하위 호환되므로 `cu124` 빌드가 정상 작동합니다.

설치 확인:
```python
import torch
print(torch.__version__)          # 예: 2.5.x+cu124
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 3060 Ti
```

---

### Step 2: 나머지 패키지

```bash
pip install ml-collections tensorboardX medpy SimpleITK
```

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `ml-collections` | 최신 | TransUNet config 관리 (`ml_collections.ConfigDict`) |
| `tensorboardX` | 최신 | TensorBoard 연동 (원본 코드 의존성) |
| `medpy` | 최신 | 의료 영상 평가 지표 (Dice, HD95) |
| `SimpleITK` | 최신 | NIfTI/NII 파일 처리 (원본 Synapse 데이터셋용) |

---

### 전체 한 번에 설치 (복붙용)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ml-collections tensorboardX medpy SimpleITK
```

---

## 사전 학습 가중치 다운로드

TransUNet은 ImageNet-21k로 사전 학습된 `R50+ViT-B_16` 가중치를 사용합니다.

### 저장 경로
```
segmentation-beta/
└── model/
    └── vit_checkpoint/
        └── imagenet21k/
            └── R50+ViT-B_16.npz   ← 이 위치에 저장
```

### 다운로드 방법

**방법 1: Google Cloud Storage (공식)**
```bash
mkdir -p ../model/vit_checkpoint/imagenet21k
cd ../model/vit_checkpoint/imagenet21k

# gsutil 사용
gsutil cp gs://vit_models/imagenet21k/R50+ViT-B_16.npz .
```

**방법 2: wget 직접 다운로드**
```bash
mkdir -p ../model/vit_checkpoint/imagenet21k
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz \
     -O ../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

> 가중치 파일이 없으면 노트북이 경고 메시지를 출력하고
> **랜덤 초기화로 학습을 계속 진행**합니다.
> (수렴 속도가 느려질 수 있음)

---

## 설치 후 최종 확인

```python
# 아래 코드 전체가 에러 없이 실행되면 준비 완료
import torch, torchvision, numpy, PIL, scipy
import tqdm, tensorboard, tensorboardX
import ml_collections, medpy, SimpleITK, h5py

print("✅ 모든 패키지 정상")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 노트북 실행 방법

```bash
cd TransUNet
jupyter lab run_training.ipynb
# 또는
jupyter notebook run_training.ipynb
```

셀 실행 순서:
1. **Section 1** (처음 1회만): 패키지 설치 → 커널 재시작
2. **Section 2**: 환경 초기화 및 CFG 설정
3. **Section 3**: 데이터셋 검증 및 시각화
4. **Section 4**: 모델 초기화
5. **Section 5**: 손실 함수 / 옵티마이저 설정
6. **Section 6**: 학습 함수 정의
7. **Section 7**: 학습 실행 ← 시간 소요
8. **Section 8**: 결과 시각화
9. **Section 9**: 파일 정리

---

## VRAM 부족 시 대처법

RTX 3060 Ti (8GB) 기준 배치 크기별 예상 VRAM:

| `IMG_SIZE` | `BATCH_SIZE` | 예상 VRAM | AMP |
|-----------|-------------|-----------|-----|
| 512 | 8 | ~7.5 GB | ✅ 필수 |
| 512 | 4 | ~4.0 GB | 권장 |
| 512 | 2 | ~2.5 GB | 선택 |
| 224 | 16 | ~4.0 GB | 권장 |

`CFG.BATCH_SIZE` 와 `CFG.USE_AMP` 값을 조정하세요.
