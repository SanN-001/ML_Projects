import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Model architecture
input_shape = x_train.shape[1:]  # Input shape based on training data
model = Sequential()

# First CNN Layer
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Dropout(0.2)))

# Second CNN Layer
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Dropout(0.3)))

# Flattening for LSTM input
model.add(TimeDistributed(Flatten()))

# LSTM Layer
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.4))

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(8, activation='softmax'))  # Assuming 8 emotion classes

# Model summary
model.summary()

# Compiling the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=4, min_lr=0.0000001)

epochs = 100
history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test), callbacks=[reduce_lr])

# Evaluating the model
test_accuracy = model.evaluate(x_test, y_test)[1] * 100
print(f"Accuracy of our model on test data: {test_accuracy:.2f}%")

# Plotting accuracy and loss
epochs_range = range(epochs)
plt.figure(figsize=(20, 6))

# Training and testing loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Testing Loss')
plt.title('Training & Testing Loss')
plt.xlabel("Epochs")
plt.legend()

# Training and testing accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Testing Accuracy')
plt.title('Training & Testing Accuracy')
plt.xlabel("Epochs")
plt.legend()

plt.show()

# Predictions on test data
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)

# Create DataFrame for predicted and actual labels
df = pd.DataFrame({'Predicted Labels': y_pred.flatten(), 'Actual Labels': y_test.flatten()})
print(df.head(10))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
cm_df = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
sns.heatmap(cm_df, annot=True, fmt='', cmap='Blues', linecolor='white', linewidth=1)
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))
