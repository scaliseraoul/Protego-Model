import os
import librosa
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

def extract_features(directory):
    features, labels = [], []
    for folder in os.listdir(directory):
        class_label = folder
        folder_path = os.path.join(directory, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            audio, sample_rate = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            features.append(mfccs_processed)
            labels.append(class_label)
    return np.array(features), np.array(labels)

def build_dnn_model(input_shape, num_classes):
    model = Sequential([
        layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    model = build_dnn_model((train_features.shape[1], 1), len(le.classes_))

    # Setup callbacks
    checkpoint = ModelCheckpoint('keras/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Train the model
    history = model.fit(
        np.expand_dims(train_features, axis=-1), train_labels,
        validation_data=(np.expand_dims(val_features, axis=-1), val_labels),
        epochs=100, batch_size=32, callbacks=[checkpoint, early_stopping])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(np.expand_dims(test_features, axis=-1), test_labels_encoded)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.2f}")

    # Predict and calculate F1 Score
    predictions = model.predict(np.expand_dims(test_features, axis=-1))
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(test_labels_encoded, predictions, average='weighted')
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a DNN-HMM hybrid model on urban sound data.')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test data')
    args = parser.parse_args()

    main(args.train_dir, args.test_dir)