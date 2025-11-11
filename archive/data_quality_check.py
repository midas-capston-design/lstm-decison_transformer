#!/usr/bin/env python3
"""
데이터 품질 체크 스크립트
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter

print("="*60)
print("📊 데이터 품질 분석 리포트")
print("="*60)

# 1. 파일 구조 분석
files = sorted([f for f in os.listdir('law_data') if f.endswith('.csv')])
print(f"\n1️⃣  파일 구조")
print(f"   총 CSV 파일: {len(files)}개")

# 노드 쌍 분석
routes = defaultdict(list)
for f in files:
    parts = f.replace('.csv', '').split('_')
    if len(parts) == 3:
        start, end, trial = parts
        routes[f"{start}→{end}"].append(trial)

print(f"   총 경로: {len(routes)}개")

# 시도 횟수 분포
trial_counts = Counter([len(trials) for trials in routes.values()])
print(f"\n   경로당 시도 횟수 분포:")
for count in sorted(trial_counts.keys()):
    print(f"     {count}회: {trial_counts[count]}개 경로")

# 2. 샘플 데이터 로드 및 분석
print(f"\n2️⃣  데이터 품질 (샘플 5개 파일)")

for i, sample_file in enumerate(files[:5], 1):
    filepath = f'law_data/{sample_file}'
    df = pd.read_csv(filepath)

    print(f"\n   [{i}] {sample_file}")
    print(f"       행 개수: {len(df):,}")
    print(f"       열: {list(df.columns)}")

    # 결측치 확인
    missing = df.isnull().sum()
    if missing.any():
        print(f"       ⚠️  결측치: {missing[missing > 0].to_dict()}")
    else:
        print(f"       ✅ 결측치 없음")

    # 지자기 데이터 통계
    mag_cols = ['MagX', 'MagY', 'MagZ']
    print(f"       지자기 범위:")
    for col in mag_cols:
        print(f"         {col}: [{df[col].min():.2f}, {df[col].max():.2f}] (평균: {df[col].mean():.2f})")

# 3. 전체 데이터 크기 및 시간 분석
print(f"\n3️⃣  전체 데이터 통계")

total_rows = 0
total_duration = []

for f in files[:20]:  # 샘플 20개
    df = pd.read_csv(f'law_data/{f}')
    total_rows += len(df)

    # 시간 차이 계산
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    duration = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds()
    total_duration.append(duration)

print(f"   평균 행 수/파일: {total_rows/20:,.0f}")
print(f"   평균 지속 시간: {np.mean(total_duration):.1f}초")
print(f"   평균 샘플링 레이트: {(total_rows/20) / np.mean(total_duration):.1f} Hz")

# 4. 노드 분포
print(f"\n4️⃣  경로 패턴 (샘플 20개)")
for route in sorted(routes.keys())[:20]:
    print(f"   {route}: {len(routes[route])}회")

print(f"\n{'='*60}")
print("✅ 데이터 품질 체크 완료!")
print(f"{'='*60}\n")
