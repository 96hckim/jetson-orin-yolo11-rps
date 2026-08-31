# TensorRPS (Jetson Embedded Engine)

NVIDIA Jetson Orin 엣지 디바이스 환경에서 실시간으로 구동되는 **TensorRT 기반 초고속 가위바위보 비전 AI 게임 엔진**입니다.
YOLOv11 커스텀 제스처 탐지 모델과 TensorRT 10.x 가속 엔진을 결합하여 초저지연(Zero-Allocation) 추론 파이프라인과 FSM 기반 게임 상태 머신, 사이버펑크 스타일의 HUD UI를 제공합니다.

---

## 🎬 Demonstration

> 💡 **Full Demo Video**: https://github.com/user-attachments/assets/56a1ff8a-05af-4022-92a5-0b77f87816be

---

## 🎮 Key Features & Game Modes

- **PvP (2인 로컬 대전 모드)**: 화면을 좌우로 분할하여 양측 플레이어의 손 제스처를 독립적으로 탐지 및 채택하여 대전을 진행합니다.

- **PvE Normal (AI 일반 대전)**: 라운드 시작 시 PC가 제스처를 랜덤하게 결정하여 대전합니다.

- **PvE God Mode (AI 무적 얌체 모드)**: 엣지 디바이스의 실시간성을 극대화한 모드로, 플레이어의 손 제스처를 실시간 판별하여 무조건 이기는 패를 즉각 대응 출제합니다.

- **안정적인 다수결 판정 (Majority Voting)**: 단일 프레임 오인식 및 노이즈 방지를 위해 10프레임 롤링 윈도우 기반 최빈값(Mode) 판정 알고리즘을 적용했습니다.

---

## ⚡ Technical Highlights

### 1. TensorRT 10.x & Zero-Allocation 추론 파이프라인

- **Pinned Memory 사전 할당**: 루프 내 동적 메모리 할당(`malloc`) 오버헤드를 제거하기 위해 Host/Device Page-Locked 버퍼를 초기화 시 1회 고정 할당합니다.

- **비동기 스트림 파이프라인**: CUDA Stream 기반으로 `Host to Device 전송` $\rightarrow$ `execute_async_v3 추론` $\rightarrow$ `Device to Host 복사`를 비동기 처리합니다.

### 2. 고속 전/후처리 최적화

- **종횡비 보존 Letterbox**: 원본 해상도 비율을 왜곡하지 않는 경량 리사이즈 및 패딩 전처리를 적용했습니다.

- **NumPy 벡터화 & OpenCV C++ NMS**: 8,400개 앵커에 대해 NumPy 기반의 클래스 신뢰도 필터링을 거친 후 `cv2.dnn.NMSBoxes`를 통해 연산 병목을 최소화했습니다.

### 3. FSM (유한 상태 머신) & Strategy Pattern 설계

- `IDLE` $\rightarrow$ `COUNTDOWN` $\rightarrow$ `JUDGING` $\rightarrow$ `ROUND_RESULT` $\rightarrow$ `MATCH_OVER`의 일관된 라이프사이클을 관리합니다.

- `BaseGameMode` 추상 클래스를 기반으로 모드별(PvP, PvE) 제스처 추출 전략을 캡슐화하여 확장성을 확보했습니다.

### 4. 고성능 인플레이스(In-Place) HUD 렌더링

- 전체 프레임 복사(`frame.copy()`)를 배제하고, 변경 영역(Sub-ROI)에만 인플레이스 `cv2.addWeighted` 알파 블렌딩을 적용하여 렌더링 레이턴시를 최소화했습니다.

---

## 📂 Project Structure

```text
├── assets/                  # UI 아이콘 리소스 (rock.png, paper.png, scissors.png)
├── configs/
│   └── app_config.py        # Vision, Camera, GamePlay 전역 불변 설정 (dataclass)
├── core/
│   ├── detector.py          # YOLOv11 전/후처리 및 엔드투엔드 탐지 파이프라인
│   └── trt_engine.py        # TensorRT 10.x CudaStream 비동기 엔진 래퍼
├── game/
│   ├── modes/
│   │   ├── base.py          # 전략 패턴 기반 BaseGameMode 추상 클래스
│   │   ├── pve.py           # PvE (일반 / God Mode) 모드 구현체
│   │   └── pvp.py           # PvP 좌우 화면 분할 모드 구현체
│   ├── rules.py             # 가위바위보 상성 및 카운터 패 순수 판정 로직
│   ├── state_machine.py     # 게임 라이프사이클 FSM 및 윈도우 다수결 판정기
│   └── types.py             # GameMode, Gesture, GameState, GameResult Enum
├── ui/
│   ├── colors.py            # BGR 팔레트 및 제스처별 테마 색상 정의
│   └── renderer.py          # HUD, 스코어보드, 바운딩 박스, 하단 티커 오버레이 렌더러
└── main.py                  # 메인 이벤트 루프 및 V4L2 카메라 캡처 엔트리포인트
```

---

## 🛠 Tech Stack & Requirements

- **Hardware**: NVIDIA Jetson Orin (Nano / NX / AGX), USB / CSI Camera
- **Language**: Python 3.10+
- **Deep Learning**: TensorRT 10.x, PyCUDA, YOLOv11
- **Vision & UI**: OpenCV (V4L2, C++ DNN), NumPy

---

## 🚀 Getting Started

### 1. Requirements Installation

```bash
# Jetson 환경 필수 패키지
pip install numpy opencv-python pycuda
```

### 2. Model Preparation

YOLOv11 기반 `.engine` 파일을 `models/` 디렉터리에 배치합니다.

```text
models/rps_yolo11n_custom_640.engine
```

### 3. Run

```bash
python main.py
```

---

## ⌨️ Controls

- **`[SPACE]`** : 새 라운드 시작 (카운트다운 진입)
- **`[1]`** : 2인 대전 모드 (PvP) 전환
- **`[2]`** : PC 일반 대전 모드 (PvE Normal) 전환
- **`[3]`** : PC 얌체 무적 모드 (PvE God Mode) 전환
- **`[R]`** : 현재 매치 스코어 초기화
- **`[Q]` / `[ESC]`** : 애플리케이션 안전 종료
