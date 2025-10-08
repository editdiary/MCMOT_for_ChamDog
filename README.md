> 아래의 내용들은 초기 README 파일 구상을 위한 탬플릿 차원에서 작성한 것이기 때문에 추후 내용이 많이 바뀔 수 있습니다.

# Multi-Camera Tracking (MCT) Project

이 프로젝트는 특정 하드웨어 시스템 상에서 동작하는 실시간 Multi-Camera Tracking (MCT) 시스템을 구현하는 것을 목표로 합니다.

## 1. 프로젝트 소개

여러 대의 카메라 시야에 들어오는 특정 객체(사람, 차량 등)를 탐지하고, 동일한 객체에 대해서는 카메라가 바뀌더라도 고유한 ID를 유지하며 지속적으로 추적하는 기술을 개발합니다. 관련 연구 논문들을 바탕으로 기술을 학습하며, 단계별로 기능을 구현해 나갑니다.

## 2. 주요 접근 방법

본 프로젝트는 MCT의 대표적인 방법론 중 하나인 **계층적 접근법 (Two-step Hierarchical Approach)**을 기반으로 개발됩니다. 이 접근법은 전체 문제를 두 개의 주요 단계로 나누어 해결합니다.

* **Step 1: Single-Camera Tracking (SCT)**
    * 각 카메라 영상에서 개별적으로 객체를 탐지(Object Detection)합니다.
    * 탐지된 객체들을 프레임 간에 연결하여 각 카메라 내에서의 짧은 궤적인 **Tracklet**을 생성합니다.

* **Step 2: Inter-Camera Association**
    * Step 1에서 생성된 Tracklet들의 외형(appearance) 및 시공간적 특징(spatio-temporal features)을 분석합니다.
    * 카메라 간 분석을 통해 동일 객체에 속하는 Tracklet들을 서로 연결(association)하여 최종적인 Global ID를 부여합니다.

이러한 계층적 접근법은 각 기능이 모듈화되어 있어 구현 및 디버깅이 용이하고, 기존에 잘 연구된 단일 카메라 추적 알고리즘 등을 활용할 수 있다는 장점이 있습니다.

## 3. 개발 로드맵 (To-Do List) - Tentative

-   [ ] **Initial: Multi-Camera capture 모듈 구현**
-   [ ] **Step 1: Single-Camera Object Detection 모듈 구현**
-   [ ] **Step 1: 검출된 객체를 연결하여 Tracklet 생성 기능 구현**
-   [ ] **Step 2: 카메라 간 Tracklet 매칭 및 연결 알고리즘 구현**
-   [ ] **전체 파이프라인 통합 및 최적화**
-   [ ] **하드웨어 시스템 배포 및 테스트**

## 4. 설치 및 실행 방법

```bash
# (추후 업데이트 예정)
# 1. 저장소 클론
git clone [Your-Repo-URL]

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. 실행
python main.py --config configs/default.yaml
```

## 5. 프로젝트 구조 (예시)
```
|-- detection/      # Step 1: Object Detection 모듈
|-- tracking/       # Step 1: Single-Camera Tracking 모듈
|-- association/    # Step 2: Inter-Camera Association 모듈
|-- configs/        # 설정 파일
|-- data/           # (gitignore 처리됨) 테스트 데이터
|-- main.py         # 메인 실행 스크립트
|-- README.md
|-- .gitignore
```