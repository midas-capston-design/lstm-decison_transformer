# 퍼지 정규화 기반 LSTM 실험

지자기·자세 센서를 **Low/Medium/High** 퍼지 멤버십으로 변환해 모델 입력을 3배 확장한 버전을 다룹니다. 극값이나 드리프트에 덜 민감한 표현을 얻는 것이 목표입니다.

## 아이디어 요약
- 센서별 백분위(25/50/75%)를 기준으로 `Low`, `Medium`, `High` 멤버십 함수를 정의
- 원본 6차원 시퀀스 → 18차원(6 × 3) 시퀀스로 확장
- “값이 크다/중간이다/작다”를 동시에 표현해 비선형 패턴을 포착

## 디렉토리 구조
```
lstm/fuzzy_normalization/
├── preprocessing_fuzzy.py        # law_data → processed_data_v3_fuzzy
├── processed_data_v3_fuzzy/      # X/y/metadata numpy + pkl
└── README.md
```

## 사용법
```bash
cd lstm/fuzzy_normalization
python preprocessing_fuzzy.py      # 퍼지 멤버십 기반 전처리
cd ..
bash train.sh 5                    # 옵션 5 = Fuzzy + LSTM 학습
```

## 출력 파일
- `processed_data_v3_fuzzy/X_train.npy` 등 분할별 데이터
- `metadata.pkl`에 사용한 퍼지 경계, 정규화 정보 저장

## 장점 & 체크포인트
1. **노이즈 강건성**: 센서값이 튀어도 멤버십이 완만하게 변함
2. **해석 가능성**: “MagX가 High”, “Pitch가 Medium” 같은 직관적 특징 제공
3. **확장성**: 다른 센서(가속도 등)에도 동일한 로직 적용 가능

향후에는 퍼지 멤버십 수 조정, Adaptive 퍼지 함수, MaNFA/DT 입력으로의 확장 등을 실험해볼 수 있습니다.
