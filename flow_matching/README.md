# Flow Matching Suite

이 폴더는 Flow Matching 기반 실내측위 실험을 위한 모든 구성요소를 담고 있습니다.

```
flow_matching/
├── model*.py, train_*.py         # 학습/추론 스크립트
├── diagnose.py 등                # 분석/검증 유틸
├── preprocessing_flow_matching*.py
├── processed_data_flow_matching/  # 윈도우 데이터 (states/coords/trajectories/labels)
└── processed_data_flow_matching_w100_backup/
```

- 전처리는 `python flow_matching/preprocessing_flow_matching_v2.py` 처럼 실행하면 되며, 생성된 결과는 동일한 폴더 안의 `processed_data_flow_matching/`에 저장됩니다.
- 학습/재현/분석 스크립트는 모두 `Path(__file__).parent` 기준 경로를 사용하므로, 저장소 어디에서 실행해도 동작합니다.
- MaNFA(`manfa/`)와 Flow Matching baseline은 동일한 데이터(`flow_matching/processed_data_flow_matching`)를 공유하니, 필요 시 전처리만 한 번 수행하면 됩니다.
