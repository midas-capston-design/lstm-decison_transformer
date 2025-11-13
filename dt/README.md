# Decision Transformer 실험 허브

Decision Transformer(DT) 계열 실험에 필요한 전처리·데이터·학습 스크립트를 모아 둔 공간입니다. LSTM 파생 데이터나 Flow Matching보다 “목표 조건 경로 생성”에 집중한 모델을 검증하고 싶을 때 이 디렉토리만 보면 됩니다.

```
dt/
├── decision_transformer/               # v1: lstm/v3 데이터 사용 (Stride 5)
├── decision_transformer_v2/            # v2: 0.9m 그리드(lstm/v4) 기반
├── decision_transformer_v2_fuzzy/      # v2 + 퍼지 정규화(lstm/v4_fuzzy)
├── preprocessing_decision_transformer.py
├── preprocessing_dt_proper.py
├── preprocessing_dt_augmented.py
├── preprocessing_dt_trajectory.py
└── processed_data_dt/                  # 공통 numpy/pkl 산출물
```

## 전처리 파이프라인
- 네 개의 전처리 스크립트는 모두 `law_data/`를 읽어 `dt/processed_data_dt/`에 `states`, `trajectories`, `rtg`, `metadata` 등을 저장합니다.
- Decision Transformer 학습 스크립트는 LSTM 전처리 결과(`lstm/v3`, `lstm/v4`, `lstm/v4_fuzzy`)를 추가로 참고합니다. 따라서 **LSTM 쪽 전처리를 먼저 실행**한 뒤 DT 전처리를 돌리면 됩니다.

## 학습 스크립트 실행 예시
```bash
# v1 (lstm/v3 데이터 기반)
python dt/decision_transformer/train_decision_transformer.py

# v2 (0.9m grid)
python dt/decision_transformer_v2/train_decision_transformer.py

# v2 + Fuzzy Normalization
python dt/decision_transformer_v2_fuzzy/train_decision_transformer.py
```

각 학습 스크립트는 `dt/processed_data_dt/` 경로를 자동으로 사용하며, 체크포인트는 저장소 루트의 `models/decision_transformer_best.pt`로 저장됩니다.

## 참고 사항
- `dt/decision_transformer/README.md`와 `dt/decision_transformer_v2/README.md`에 모델별 세부 아이디어와 하이퍼파라미터가 정리되어 있습니다.
- MaNFA/Flow Matching과 동일한 노드 그래프·센서 구성을 사용하므로 라벨 일관성이 유지됩니다.
- 실험 후 산출물을 다른 스택과 비교하고 싶다면 `models/` 디렉토리와 `results/` 폴더를 함께 관리하세요.

필요한 전처리 → 학습 순서를 이 디렉토리 안에서 모두 해결할 수 있도록 만들어 두었으니, Decision Transformer 관련 실험은 여기서 이어가면 됩니다.
