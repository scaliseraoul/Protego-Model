import os
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set directories
train_dir = 'train-neutral-4-trimmed-0'
test_dir = 'test-neutral-4-trimmed-0'

# Function to load PCM 16 audio files from subfolders
def load_audio_files(directory):
    audio_data = []
    labels = []
    class_names = os.listdir(directory)
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(class_path, file_name)
                    sample_rate, data = wavfile.read(file_path)
                    audio_data.append(data)
                    labels.append(class_indices[class_name])
    return audio_data, labels, class_indices

# Load training data
train_audio, train_labels, class_indices = load_audio_files(train_dir)

# Split training data into training and validation sets (80/20 split)
train_audio, val_audio, train_labels, val_labels = train_test_split(train_audio, train_labels, test_size=0.2, random_state=42)

# Load testing data
test_audio, test_labels, _ = load_audio_files(test_dir)

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# Normalize the audio data
train_audio = np.array(train_audio) / 32768.0
val_audio = np.array(val_audio) / 32768.0
test_audio = np.array(test_audio) / 32768.0

# Expand dimensions to match input shape (samples, timesteps, 1)
train_audio = np.expand_dims(train_audio, axis=-1)
val_audio = np.expand_dims(val_audio, axis=-1)
test_audio = np.expand_dims(test_audio, axis=-1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(train_audio.shape[1], 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Setup callbacks
checkpoint = ModelCheckpoint(f'keras/{train_dir}-model-pcm.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Train the model with validation data
history = model.fit(train_audio, train_labels, epochs=10, validation_data=(val_audio, val_labels), callbacks=[checkpoint, early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_audio, test_labels)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
print(f'Test loss: {test_loss * 100:.2f}%')

# Predict and calculate F1 Score
predictions = model.predict(test_audio)
predicted_labels = np.argmax(predictions, axis=1)
f1 = f1_score(test_labels, predicted_labels, average='weighted')
print(f'F1 Score: {f1:.2f}')

# Save results to CSV
def save_results_to_csv(base_dir, iterations, test_accuracy, test_loss, f1):
    data = {
        'Base Directory': base_dir,
        'Iterations Before Convergence': iterations,
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'F1 Score': f1
    }
    df = pd.DataFrame([data])
    with open('results.csv', 'a', newline='') as file:
        df.to_csv(file, index=False, header=file.tell()==0)

save_results_to_csv(train_dir, len(history.history['loss']), test_accuracy, test_loss, f1)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.tight_layout()
plt.savefig(f'images/{train_dir}_plot.png')
plt.close()

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('audio_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)
