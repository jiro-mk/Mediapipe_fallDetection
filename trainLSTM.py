from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import glob



# MP推定データのパス
DATA_PATH = os.path.join('MP_Data') 

# 推論するアクション
actions = np.array(['fall','standing'])

# Thirty videos worth of data
DATASET_PATH = os.path.join('data/*') 
seq_files=[]

for name in sorted(glob.glob(DATASET_PATH)):
    seq_files.append(name)

# 各アクションのフォルダ数
no_sequences = 14

# 1アクションのフレーム数
sequence_length = 30

# データセット作成
import glob
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            a = glob.glob(os.path.join(DATA_PATH, action, str(sequence))+'/*.npy')
            # print(a)
            if len(a)==0:
                continue
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# LSTMモデル定義
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))



# optimizer = adam_v2.Adam(lr=0.001)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 学習データ、テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

# 学習スタート
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
model.summary()

# モデルの保存
model.save('action.h5')

