import os
import librosa
import numpy as np
from scipy import signal
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, Dropout, Flatten, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

start_time = time.time()


def extract_features(file_path, max_pad_len=216):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Apply high-pass filter for noise reduction
    sos = signal.butter(10, 80, 'hp', fs=sample_rate, output='sos')
    audio = signal.sosfilt(sos, audio)

    # Extract MFCC and normalize
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))

    # Pitch extraction and NaN handling
    pitch, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nan_to_num(pitch)
    pitch = (pitch - np.min(pitch)) / (np.max(pitch) - np.min(pitch)) if np.max(pitch) != 0 else pitch

    # Padding
    pad_width = max(0, max_pad_len - mfccs.shape[1])
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    pitch = np.pad(pitch, (0, pad_width), mode='constant')
    feature = np.concatenate((mfccs, pitch.reshape(1, -1)), axis=0)
    return feature


data_path = 'C:/Users/sanan/ML_Projects/Emotion_Recognition'
X, y = [], []
emotion_dict = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful',
                '07': 'disgust', '08': 'surprised'}

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                emotion = emotion_dict[file.split('-')[2]]
                feature = extract_features(os.path.join(folder_path, file))
                X.append(feature)
                y.append(emotion)

X = np.array(X)
y = np.array(y)
y = to_categorical([list(emotion_dict.values()).index(em) for em in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                          input_shape=(None, X_train.shape[2], X_train.shape[3], 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 1))))

model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 1))))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test_acc * 100)
print("Total Time:", time.time() - start_time)