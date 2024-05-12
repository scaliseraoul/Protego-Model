import os
import librosa
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

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

def save_results_to_csv(base_dir, iterations, test_accuracy, test_loss, y_true, y_pred, f1):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='Path to train directory')
    parser.add_argument('--test_dir', type=str, help='Path to test directory')
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir

    data_features, data_labels = extract_features(train_dir)
    
    # Resample the dataset
    #sampler = SMOTE()  # Default to SMOTE
    #data_features, data_labels = sampler.fit_resample(data_features, data_labels)

    print("After resampling, class distribution:")
    unique, counts = np.unique(data_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    test_features, test_labels = extract_features(test_dir)
    train_features, val_features, train_labels, val_labels = train_test_split(
        data_features, data_labels, test_size=0.2, random_state=42)

    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    val_labels_encoded = le.transform(val_labels)
    test_labels_encoded = le.transform(test_labels)

    train_features = np.expand_dims(train_features, axis=1)
    val_features = np.expand_dims(val_features, axis=1)
    test_features = np.expand_dims(test_features, axis=1)

    model = tf.keras.Sequential([
        layers.LSTM(128, input_shape=(train_features.shape[1], train_features.shape[2])),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(le.classes_), activation='softmax')
    ])
    

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f'keras/{train_dir}.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    history = model.fit(
        train_features,
        train_labels_encoded,
        epochs=100,
        validation_data=(val_features, val_labels_encoded),
        callbacks=[checkpoint, early_stopping]
    )

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

    test_loss, test_accuracy = model.evaluate(test_features, test_labels_encoded)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss * 100:.2f}%")

    # Calculate F1 Score
    test_predictions = model.predict(test_features)
    test_predictions = np.argmax(test_predictions, axis=1)
    f1 = f1_score(test_labels_encoded, test_predictions, average='weighted')

    # Save results to CSV
    save_results_to_csv(train_dir, len(history.history['loss']), test_accuracy, test_loss, test_labels_encoded, test_predictions, f1)