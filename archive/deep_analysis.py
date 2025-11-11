#!/usr/bin/env python3
"""
심층 분석: 중복 구간에서 지자기 패턴이 실제로 얼마나 유사한지 확인
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*70)
print("🔬 심층 데이터 분석")
print("="*70)

# 1. 경로별 지자기 패턴 시각화
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

routes_to_analyze = [
    ('law_data/1_11_1.csv', '1→11', 'blue'),
    ('law_data/2_12_1.csv', '2→12', 'red'),
    ('law_data/11_1_1.csv', '11→1', 'green'),
]

print("\n📊 경로별 지자기 패턴 분석:\n")

for filepath, label, color in routes_to_analyze:
    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath)

    print(f"[{label}]")
    print(f"  샘플 수: {len(df)}")
    print(f"  지속시간: {(pd.to_datetime(df['Timestamp'].iloc[-1]) - pd.to_datetime(df['Timestamp'].iloc[0])).total_seconds():.1f}초")
    print(f"  평균 속도: {len(df) / (pd.to_datetime(df['Timestamp'].iloc[-1]) - pd.to_datetime(df['Timestamp'].iloc[0])).total_seconds():.1f} Hz")

    # MagX, MagY, MagZ 플롯
    axes[0].plot(df['MagX'].values, label=f'{label}', color=color, alpha=0.7)
    axes[1].plot(df['MagY'].values, label=f'{label}', color=color, alpha=0.7)
    axes[2].plot(df['MagZ'].values, label=f'{label}', color=color, alpha=0.7)

    print(f"  MagX: [{df['MagX'].min():.2f}, {df['MagX'].max():.2f}] μT")
    print(f"  MagY: [{df['MagY'].min():.2f}, {df['MagY'].max():.2f}] μT")
    print(f"  MagZ: [{df['MagZ'].min():.2f}, {df['MagZ'].max():.2f}] μT")
    print()

axes[0].set_ylabel('MagX (μT)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel('MagY (μT)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel('MagZ (μT)', fontsize=12)
axes[2].set_xlabel('Sample Index', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('magnetic_patterns.png', dpi=150, bbox_inches='tight')
print("✅ 저장: magnetic_patterns.png\n")

# 2. 문제 정의 분석
print("="*70)
print("💡 문제 정의 옵션 분석")
print("="*70)

options = """
현재 데이터: 69개 경로 × 평균 2회 = 187개 파일

옵션 1: ❌ 경로 분류 (Route Classification)
  - 목표: 지자기 시퀀스 → 어느 경로인가? (1→11 vs 2→12 ...)
  - 문제: 중복 구간에서 라벨 모호성
  - 클래스 수: 69개
  - 결론: 불가능

옵션 2: ✅ 노드 위치 예측 (Node Localization) - LSTM
  - 목표: 지자기 시퀀스 → 현재 어느 노드에 있는가?
  - 장점: 경로와 무관하게 위치만 예측
  - 클래스 수: 29개 노드
  - 방법: LSTM + Softmax
  - 문제: 중복 구간에서도 현재 노드는 동일하므로 OK!

옵션 3: ✅ 다음 노드 예측 (Next Node Prediction) - Decision Transformer
  - 목표: 현재 상태 + 목표 → 다음 어디로 갈 것인가?
  - 장점: Sequential decision making
  - 방법: Transformer + Return-to-go conditioning
  - 입력: (R̂_t, s_t, a_t) 시퀀스
    - R̂_t: 목표까지 남은 거리 (return-to-go)
    - s_t: 현재 지자기 상태 [MagX, MagY, MagZ]
    - a_t: 다음 노드로의 action
  - 출력: 다음 action (다음 노드)

옵션 4: ✅ 궤적 모델링 (Trajectory Modeling)
  - 목표: 전체 경로를 시퀀스로 모델링
  - 방법: Seq2Seq, Transformer
"""

print(options)

# 3. 권장 접근법
print("\n" + "="*70)
print("🎯 권장 접근법")
print("="*70)

recommendation = """
Phase 1: LSTM 기반 노드 위치 예측 (Baseline)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  문제 재정의:
    • 입력: 지자기 시퀀스 [MagX, MagY, MagZ] × 20 timesteps
    • 출력: 29개 노드 중 현재 위치
    • 라벨링 방법:
      1. 각 경로 파일의 중간 구간 샘플링
      2. 경로 정보(1→11)를 버리고
      3. 실제 통과한 노드만 라벨로 사용

  데이터 전처리:
    1. 각 파일을 시간 순으로 N등분 (예: 29등분)
    2. 각 구간을 해당 노드로 라벨링
    3. Sliding window로 시퀀스 생성

  장점:
    ✓ 중복 구간 문제 해결 (같은 노드 = 같은 라벨)
    ✓ 단순하고 검증 가능
    ✓ 빠른 구현

Phase 2: Decision Transformer (Advanced)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  문제 정의:
    • 목표: 시작→목표 최단 경로 찾기
    • 입력: (return-to-go, state, action) 시퀀스
    • 출력: 다음 action (다음 노드)

  Return-to-go 정의:
    • 목표 노드까지 남은 거리 (m)
    • 또는 남은 노드 개수

  State 정의:
    • 지자기: [MagX, MagY, MagZ]
    • 현재 노드 ID (optional)

  Action 정의:
    • 다음 방문할 노드 ID

  장점:
    ✓ Context-aware (시작점 고려)
    ✓ Goal-conditioned (목표 명시)
    ✓ 최적 경로 학습 가능
"""

print(recommendation)

print("\n" + "="*70)
print("🔧 다음 단계")
print("="*70)
print("""
1. Phase 1 데이터 전처리 시작
   → 경로 파일을 노드별로 분할하여 라벨링

2. LSTM 모델 구축 및 학습
   → Baseline 성능 확인

3. (Optional) Phase 2 구현
   → Decision Transformer 비교

어떤 단계부터 시작할까요?
""")
