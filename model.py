# emotion_detection/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import clean_text
import numpy as np
import pickle
import os

tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'text_emotion_detection', 'tokenizer.pkl')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


def build_model(input_dim, output_dim=6):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=16, input_length=100))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=6, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])
        
    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model2.weights.h5")
    return history

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test)

def predict_emotion(model, input_text,label_mapping):
    cleaned_text = clean_text(input_text)
    cleaned_text = cleaned_text.replace("http", "").replace("href", "").replace("img", "").replace("irc", "")

    input_sequence = tokenizer.texts_to_sequences([cleaned_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=100, padding='post')
    predictions = model.predict(input_sequence_padded)
    predicted_class_index = np.argmax(predictions[0])
    for label, value in label_mapping.items():
        if value == predicted_class_index:
            return label

