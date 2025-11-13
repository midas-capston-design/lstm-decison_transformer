# MaNFA (Magnetic Neural Field Alignment)

지자기 시계열로 실내 위치를 추정하기 위해 **Neural Field + Mamba 기반 SSM + 입자 플로우**를 결합한 연구용 프로토타입입니다. 250샘플/stride 50 윈도우, 6개 신뢰 채널(MagX/Y/Z, Pitch/Roll/Yaw)을 그대로 사용하며, `(x, y)` 연속 좌표를 직접 출력합니다.

## 구성
```
manfa/
├── README.md                # 사용 안내
├── requirements.txt         # PyTorch + mamba-ssm 등 의존성
├── configs/                 # 하이퍼파라미터 YAML(기본/H100)
├── config.py                # dataclass 기반 설정 파서
├── data/
│   ├── quality.py           # 아크비율/분산/야우 상관 검증
│   └── window_dataset.py    # processed_data 로더 + 캐시
├── models/                  # Neural Field, Mamba Encoder, Particle Flow
├── train.py                 # Stage별 학습 스크립트(field/full)
├── inference.py             # 스트리밍 추론기
└── run_manfa*.sh            # 파이프라인 실행 스크립트
```

## 빠른 시작
1. 가상환경 준비
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r manfa/requirements.txt
   ```

2. 스테이지 실행 (8 GB GPU)
   ```bash
   ./manfa/run_manfa.sh
   ```
   - Stage 1: Neural Field + Encoder 사전학습(`results/manfa_field/`)
   - Stage 2: Particle Flow 포함 전체 학습(`results/manfa_full/`)
   - Stage 3: `flow_matching/processed_data_flow_matching` 기준 test 추론

3. 대용량 GPU(H100 등) 구성
   ```bash
   ./manfa/run_manfa_h100.sh
   ```
   더 큰 Mamba 차원·입자 수를 사용하며 나머지 절차는 동일합니다.

## 데이터 품질 체크
`manfa/data/quality.py`는 학습 전 창마다 아래 조건을 확인합니다.
1. **Arc/Net 비율**: 누적 이동 거리 ÷ 직선 거리 > 임계치 → 라벨 불량으로 판단
2. **자기장 분산**: 센서 멈춤 여부 파악
3. **Yaw-자기장 상관**: 방향 전환 시 MagX/MagY 위상이 함께 움직이는지 확인

임계값은 YAML(config.data)에 있으며, 통과한 인덱스를 `processed_data_flow_matching/clean_indices_*.json`에 캐시합니다.

## 손실 구성
- `λ_field` · SmoothL1: Neural Field가 관측 시퀀스를 재구성하도록 학습
- `λ_contrastive` · InfoNCE: 관측 latent와 필드 latent 정렬
- `λ_particle` · SmoothL1: 입자 posterior의 평균을 실제 `(x, y)`에 맞춤 + 엔트로피 규제

모든 가중치는 `manfa/configs/*.yaml`에서 바로 수정 가능합니다.

## 추론 흐름
1. 새 250샘플 윈도우를 로드하고 품질 필터를 통과시킵니다.
2. Mamba 인코더가 관측 latent를 계산합니다.
3. Neural Field가 후보 좌표에서 가상 센서 시퀀스를 생성하고, particle flow가 잔차 기반으로 입자를 이동합니다.
4. 최종 posterior 평균이 실측 단위 `(x, y)`로 반환되며, stride 50 주기로 갱신됩니다.

## 다음 단계 제안
- `manfa/train.py --stage field/full` 조합으로 하이퍼파라미터 스윕
- `models/`에 저장된 Flow Matching 체크포인트와 성능 비교
- 입자 샘플러(`ParticleInitializer`)에 노드 그래프 priors 또는 보행자 관성 정보를 추가

필요한 부분은 자유롭게 수정해서 사용하고, 실험 로그/체크포인트는 `results/` 폴더에 쌓아두면 관리가 편합니다.
