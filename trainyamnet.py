import os
import librosa
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt

def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, audio is already 16 kHz"""
    # Load the audio file with librosa
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

def extract_features(directory):
  features, labels = [], []
  yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
  yamnet_model = hub.load(yamnet_model_handle)

  for folder in os.listdir(directory):
    class_label = folder
    folder_path = os.path.join(directory, folder)
    for file in os.listdir(folder_path):
      file_path = os.path.join(folder_path, file)
      # Load and pre-process audio (ensure it matches YAMNet requirements)
      testing_wav_data = load_wav_16k_mono(file_path)
      # Add batch dimension
      scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
      features.append(embeddings.numpy())  # Extract embedding from 1st dim
      labels.append(class_label)
  return np.array(features), np.array(labels)

def build_dnn_model(input_shape):
    print(input_shape)
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        layers.LSTM(128,return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def main(train_dir, test_dir):
    # Load data
    data_features, data_labels = extract_features(train_dir)
    test_features, test_labels = extract_features(test_dir)
    
    # Encode labels
    le = LabelEncoder()
    data_labels_encoded = le.fit_transform(data_labels)
    test_labels_encoded = le.transform(test_labels)

    # Split data into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        data_features, data_labels_encoded, test_size=0.2, random_state=42)
    # Build the DNN model

    model = build_dnn_model((None, train_features.shape[2]))

    # Setup callbacks
    checkpoint = ModelCheckpoint(f'keras/{train_dir}-yammnet.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Add class weight
    # layers.BatchNormalization()
    # Train the model
    history = model.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=100, batch_size=32, callbacks=[checkpoint, early_stopping])

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
    plt.savefig(f'images/{train_dir}_yammnet_plot.png')
    plt.close()


    # Evaluate the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_features, test_labels_encoded)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss * 100:.2f}%")
    print(f"Test AUC: {test_auc:.2f}")

    # Predict and calculate F1 Score
    predictions = model.predict(test_features)
    predictions = (predictions > 0.5).astype(int).flatten()
    f1 = f1_score(test_labels_encoded, predictions, average='weighted')
    print(f"F1 Score: {f1:.2f}")
    save_results_to_csv(train_dir, len(history.history['loss']), test_accuracy, test_loss, f1, test_auc)

def save_results_to_csv(base_dir, iterations, test_accuracy, test_loss, f1, test_auc):
    data = {
        'Base Directory': base_dir,
        'Iterations Before Convergence': iterations,
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'F1 Score': f1,
        'AUC': test_auc
    }
    df = pd.DataFrame([data])
    with open('results.csv', 'a', newline='') as file:
        df.to_csv(file, index=False, header=file.tell()==0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a DNN-HMM hybrid model on urban sound data.')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test data')
    args = parser.parse_args()

    main(args.train_dir, args.test_dir)