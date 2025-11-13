#!/usr/bin/env python3
"""
지자기 기반 실내 위치 추정 - LSTM 모델 학습
"""
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print("="*70)
print("🚀 LSTM 모델 학습 시작 (v4 + Fuzzy)")
print("="*70)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1/5] 데이터 로드...")

data_dir = Path('processed_data_v4_fuzzy')

X_train = np.load(data_dir / 'X_train.npy')
y_train = np.load(data_dir / 'y_train.npy')

X_val = np.load(data_dir / 'X_val.npy')
y_val = np.load(data_dir / 'y_val.npy')

X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

with open(data_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

num_classes = metadata['num_classes']

print(f"  Train: {X_train.shape} → {num_classes} 클래스")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  클래스 수: {num_classes}")

# ============================================================================
# 2. 모델 정의
# ============================================================================
print("\n[2/5] 모델 구축...")

def build_lstm_model(input_shape, num_classes):
    """
    4층 LSTM 모델

    Args:
        input_shape: (timesteps, features) = (100, 6)
        num_classes: 출력 클래스 수
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # LSTM Layer 1
        layers.LSTM(128, return_sequences=True, name='lstm_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # LSTM Layer 2
        layers.LSTM(256, return_sequences=True, name='lstm_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # LSTM Layer 3
        layers.LSTM(256, return_sequences=True, name='lstm_3'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # LSTM Layer 4
        layers.LSTM(128, return_sequences=False, name='lstm_4'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    return model

# 모델 생성
input_shape = (X_train.shape[1], X_train.shape[2])  # (100, 6)
model = build_lstm_model(input_shape, num_classes)

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 요약
model.summary()

print(f"\n  총 파라미터: {model.count_params():,}")

# ============================================================================
# 3. 콜백 설정
# ============================================================================
print("\n[3/5] 학습 설정...")

# 모델 저장 디렉토리
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

callbacks = [
    # 최고 성능 모델 저장
    ModelCheckpoint(
        filepath=str(model_dir / 'lstm_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Early stopping
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        mode='max',
        verbose=1,
        restore_best_weights=True
    ),

    # Learning rate 감소
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print("  Callbacks:")
print("    - ModelCheckpoint: val_accuracy 최고 모델 저장")
print("    - EarlyStopping: patience=15")
print("    - ReduceLROnPlateau: patience=5, factor=0.5")

# ============================================================================
# 4. 모델 학습
# ============================================================================
print("\n[4/5] 모델 학습...")

BATCH_SIZE = 128
EPOCHS = 100

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 5. 학습 결과 시각화
# ============================================================================
print("\n[5/5] 학습 결과 저장...")

# 학습 곡선 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("  학습 곡선 저장: training_history.png")

# 최종 평가
print("\n[최종 평가]")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")

# 학습 히스토리 저장
with open(model_dir / 'history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# 최종 모델 저장
model.save(model_dir / 'lstm_final.keras')
print(f"\n  모델 저장 완료: {model_dir}/")

print("\n" + "="*70)
print("✅ 학습 완료!")
print("="*70)
print(f"""
📊 최종 결과:
  Train Accuracy: {train_acc*100:.2f}%
  Val Accuracy:   {val_acc*100:.2f}%
  Test Accuracy:  {test_acc*100:.2f}%

  모델 위치: {model_dir}/lstm_best.keras

다음 단계: 성능 평가 및 위치 오차 분석 🎯
""")
