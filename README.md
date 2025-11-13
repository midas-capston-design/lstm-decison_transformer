# 지자기 기반 실내 위치 추정 워크스페이스

지금 저장소는 모델 계열별(LSTM, Decision Transformer, Flow Matching, MaNFA)로 폴더를 정리해 전처리·데이터·학습 스크립트를 한곳에 모았습니다. 어떤 접근법을 실험하더라도 필요한 모든 코드와 산출물이 해당 디렉토리 안에 있습니다.

## 디렉토리 구조

```
.
├── law_data/                  # 원본 CSV (노선별 센서 기록)
├── nodes_final.csv            # 29개 노드 / 5m 연결 그래프
├── flow_matching/             # Flow Matching 베이스라인 + 250×6 윈도우 데이터
├── dt/                        # Decision Transformer 전처리/학습 코드 및 데이터
├── lstm/                      # LSTM v1~v4, 퍼지 노말라이제이션, train.sh
├── manfa/                     # Magnetic Neural Field Alignment (현재 주력)
├── models/                    # 공용 체크포인트
├── results/                   # 학습 로그, 추론 결과
└── README.md                  # 이 문서
```

### Flow Matching 스택
- `flow_matching/preprocessing_flow_matching_v2.py`: 노드 그래프 기반 250샘플(5초) 윈도우 생성, stride=50. 결과는 `flow_matching/processed_data_flow_matching/`.
- `flow_matching/train_flow_matching.py`, `model.py`: 좌표 생성 Flow Matching 모델.
- `flow_matching/tools/`, `flow_matching/plots/`: 노드 지도·데이터 분포·판별력 체크 유틸.

### Decision Transformer 스택 (`dt/`)
- 전처리 스크립트가 `law_data/`를 읽어 `dt/processed_data_dt/`에 저장.
- 학습 스크립트(`dt/decision_transformer*/train_decision_transformer.py`)는 LSTM 전처리 결과(`lstm/v3`, `lstm/v4` 등)를 참조.
- 자세한 설명은 `dt/README.md` 참고.

### LSTM 베이스라인 (`lstm/`)
- `v1`~`v4`, `v4_fuzzy`, `fuzzy_normalization`, 통합 학습 스크립트 `lstm/train.sh` 포함.
- `bash lstm/train.sh <옵션>`으로 가상환경 활성화 → 전처리 → 학습을 자동 실행.

### MaNFA (Neural Field + Mamba + Particle Flow)
- 최신 연구 방향은 `manfa/`에 정리되어 있으며 입력은 `flow_matching/processed_data_flow_matching/`을 공유합니다.
- 8 GB GPU 실행: `./manfa/run_manfa.sh`
- H100/대용량 GPU 실행: `./manfa/run_manfa_h100.sh`
- 두 스크립트 모두 필드 사전학습 → 전체 학습 → 추론까지 한 번에 수행하며 최신 체크포인트를 자동 선택합니다.

## 전처리 요약
- **원본 수집**: `law_data/*.csv` (Timestamp + MagXYZ + Pitch/Roll/Yaw + Highlighted 마커)
- **그래프 기반 윈도우**: `python flow_matching/preprocessing_flow_matching_v2.py` → `flow_matching/processed_data_flow_matching/`
- **Decision Transformer용**: `python dt/preprocessing_decision_transformer.py` → `dt/processed_data_dt/`
- **LSTM용**: `python lstm/v3/preprocessing_v3.py` 등 → `lstm/v*/processed_data_*`

모든 전처리는 동일한 노드 그래프(`nodes_final.csv`)를 사용하므로, 어느 모델을 돌려도 건물 좌표계가 일관됩니다.

## 학습/추론 진입점

| 스택 | 명령어 | 비고 |
|------|--------|------|
| MaNFA (기본) | `./manfa/run_manfa.sh` | `flow_matching/processed_data_flow_matching` 사용 |
| MaNFA (H100) | `./manfa/run_manfa_h100.sh` | 더 큰 SSM, particle 수 확대 |
| Flow Matching | `python flow_matching/train_flow_matching.py` | MaNFA와 동일한 입력 |
| Decision Transformer v1 | `python dt/decision_transformer/train_decision_transformer.py` | `lstm/v3/processed_data_v3` 필요 |
| Decision Transformer v2/fuzzy | `python dt/decision_transformer_v2/train_decision_transformer.py` 등 | `lstm/v4`, `lstm/v4_fuzzy` 필요 |
| LSTM 베이스라인 | `bash lstm/train.sh <1~7>` | v1~v4, Fuzzy, DT 연계 학습 |

## 노드 그래프 요약
- 29개 노드, 5 m 이내 연결. (10↔28), (24↔25)는 벽으로 인해 제거.
- 모든 경로는 축 정렬(kid). 시각화: `flow_matching/node_graph_map.png`.

```
           29
           |
    21-20-…-2-1
          |    |
          27   22
          |    |
   26-25-24    23
```

필요하면 `flow_matching/tools/draw_node_map.py` 또는 `test_node_graph.py`로 그래프를 다시 확인할 수 있습니다.

## 다음 단계
- MaNFA 파이프라인으로 최신 모델을 학습하고, `models/`에 있는 Flow Matching 체크포인트와 비교해 성능을 점검하세요.
- `law_data/`는 원본 보관소로 유지하고, 센서 교정/그래프 설정이 바뀌면 전처리를 다시 수행하면 됩니다.

각 서브 README(`manfa/README.md`, `flow_matching/README.md`, `dt/README.md`, `lstm/README.md`)에 세부 설명이 있으니 참고하거나 필요한 내용을 추가해 주세요.
