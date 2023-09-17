import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Step 1: Load Data from Excel
df = pd.read_csv('data.csv', usecols=['password', 'strength'])

# Clean the 'strength' column by removing any non-integer values
df['strength'] = df['strength'].astype(int)
df = df[df['strength'] != -1]

df['password'] = df['password'].astype(str)
passwords = df['password'].tolist()
labels = df['strength'].tolist()
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+"

def password_to_one_hot(password, max_length=20):
    one_hot = np.zeros((max_length, len(charset)), dtype=np.float32)
    for i, char in enumerate(password):
        if i >= max_length:
            break
        if char in charset:
            one_hot[i, charset.index(char)] = 1.0
    return one_hot

# Preprocess passwords
preprocessed_passwords = []
for password in passwords:
    preprocessed_password = password_to_one_hot(password)
    preprocessed_passwords.append(preprocessed_password)

# Step 3: Design Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(20, len(charset)), return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 output classes (0, 1, 2)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to one-hot encoding for multi-class classification
labels = to_categorical(labels, num_classes=3)  # Assuming 3 classes (0, 1, 2)

# Split Data and Train Model
X_train, X_test, y_train, y_test = train_test_split(preprocessed_passwords, labels, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Save the trained model
model.save("password_strength_model.keras")
