# Importing necessary libraries
import pandas as pd
import numpy as np
import os
import random
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Flatten, Dropout, Conv1D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau

# Setting random seed for reproducibility
import tensorflow as tf
tf.random.set_seed(30)

# Data Preparation
# Define the directory containing audio files
audio_dir = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
files = os.listdir(audio_dir)

# Prepare lists to hold our data
emotions = []
statements = []
file_paths = []

# Loop through each actor's folder and get file information
for folder in files:
    actor_files = os.listdir(os.path.join(audio_dir, folder))
    for file in actor_files:
        parts = file.split('.')[0].split('-')
        emotions.append(int(parts[2]))
        statements.append(int(parts[4]))
        file_paths.append(os.path.join(audio_dir, folder, file))

# Create a DataFrame for our audio data
data_df = pd.DataFrame({"Emotions": emotions, "Statement": statements, "Path": file_paths})

# Map numerical emotion values to actual emotion labels
emotion_map = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
data_df['Emotions'] = data_df['Emotions'].replace(emotion_map)

# Visualize emotion distribution
plt.title('Count of Emotions')
sns.countplot(data=data_df, x='Emotions')
plt.ylabel('Count')
plt.xlabel('Emotions')
plt.show()

# Function to create waveplot
def create_waveplot(data, sr):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot')
    librosa.display.waveshow(data, sr=sr)
    plt.show()

# Function to create spectrogram
def create_spectrogram(data, sr):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram')
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

# Test audio visualization with a specific emotion
def visualize_emotion(emotion):
    path = np.array(data_df.Path[data_df.Emotions == emotion])[1]
    data, sr = librosa.load(path)
    create_waveplot(data, sr)
    create_spectrogram(data, sr)
    return path  # returning path to play audio

# Visualize a few emotions
for emotion in ['fear', 'angry', 'sad', 'happy']:
    visualize_emotion(emotion)

# Data Augmentation Functions
def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def time_stretch(data, rate=0.85):
    return librosa.effects.time_stretch(data, rate)

def shift_audio(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch_shift(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr, pitch_factor)

# Feature Extraction
def extract_features(data):
    mfcc = librosa.feature.mfcc(y=data, sr=22050)
    return mfcc

def transform_audio(data, transformations):
    fn = random.choice(transformations)
    if fn == pitch_shift:
        return fn(data, 22050)
    elif fn == "None":
        return data
    else:
        return fn(data)

# Get features for each audio file
def get_audio_features(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    transformations = [add_noise, pitch_shift, "None"]

    # Get features with transformations
    features = []
    for _ in range(3):
        transformed_data = transform_audio(data, transformations)
        features.append(extract_features(transformed_data)[:, :108])  # Taking first 108 features
    return features

# Prepare data for model
X, Y = [], []
for path, emotion in zip(data_df.Path.to_list(), data_df.Emotions.to_list()):
    audio_features = get_audio_features(path)
    for feature in audio_features:
        if feature.shape == (20, 108):
            X.append(feature)
            Y.append(emotion)

X = np.array(X)
Y = np.array(Y)

# One-hot encode labels
encoder = OneHotEncoder()
Y = encoder.fit_transform(Y.reshape(-1, 1)).toarray()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21, shuffle=True)

# Reshape for LSTM input
x_train = np.expand_dims(x_train, axis=3)
x_train = np.swapaxes(x_train, 1, 2)

x_test = np.expand_dims(x_test, axis=3)
x_test = np.swapaxes(x_test, 1, 2)

# Model Building
model = Sequential()
model.add(TimeDistributed(Conv1D(16, 3, padding='same', activation='relu'), input_shape=(108, 162, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))  # 8 output classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001)
history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

# Evaluate the model
test_accuracy = model.evaluate(x_test, y_test)[1] * 100
print(f"Test accuracy: {test_accuracy:.2f}%")
