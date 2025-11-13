# Decision Transformer v2 (0.9 m Grid)

v2 모델은 `lstm/v4`에서 생성한 0.9 m 그리드 데이터와 더 큰 context를 사용해 경로를 생성하는 변형입니다. 입력·출력 구조는 v1과 동일하지만, 그리드 해상도가 내려가면서 return-to-go와 action 분포가 안정적이라 학습 수렴이 빠릅니다.

## 데이터 의존성
- **필수 전처리**  
  1. `python lstm/v4/preprocessing_v4.py`  
  2. `python dt/preprocessing_dt_proper.py` (또는 필요에 따라 augmented/trajectory 버전)
- 전처리 결과는 `lstm/v4/processed_data_v4/`와 `dt/processed_data_dt/`에 저장되고, 학습 스크립트가 자동으로 참조합니다.

## 모델 및 학습
- Transformer 블록 수·임베딩 차원은 v1과 동일하되, action을 “다음 그리드 좌표”로 바로 예측합니다.
- 학습 실행:
  ```bash
  cd dt/decision_transformer_v2
  python train_decision_transformer.py
  ```
- 체크포인트는 `models/decision_transformer_best.pt`로 저장됩니다(다른 버전과 공유하므로 필요하면 이름을 바꿔 보관하세요).

## 변경점 요약
| 항목 | v1 | v2 |
|------|----|----|
| 데이터 | lstm/v3 (Stride=5) | lstm/v4 (0.9 m grid) |
| Label | 연속 좌표 | Grid index → 좌표 |
| Noise | 상대적으로 큼 | 단순 + 균일 |
| 적용처 | 세밀한 경로 | 넓은 범위 추적 |

## TODO
- v1 대비 top-k 정확도/경로 오류 비교
- Grid size 0.9 m → 0.6 m/1.2 m 변형 실험
- Return-to-go 계산을 세그먼트 길이 기반으로 개선

Decision Transformer 논문의 아이디어를 그대로 가져오되, 실내측위 데이터 분포에 맞춰 loss/전처리를 튜닝한 버전입니다. 추가 실험 시 이 README를 기반으로 메모를 이어 주세요.
