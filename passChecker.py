import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("password_strength_model.keras")


# Function to preprocess the password (replace with your preprocessing logic)
def preProcessPassword(password, max_length=20):
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+"
    print("Processing password:", password)
    one_hot = np.zeros((max_length, len(charset)), dtype=np.float32)
    for i, char in enumerate(password):
        if i >= max_length:
            break
        if char in charset:
            one_hot[i, charset.index(char)] = 1.0
    print("one_hot: ", one_hot)
    return one_hot


# Function to check password strength
def check_password_strength(password):
    # Preprocess the password as you did during training
    preprocessed_password = preProcessPassword(password)
    # Make a prediction using the model
    prediction = model.predict(np.array([preprocessed_password]))
    print("Model prediction:", prediction)

    # Return the result
    if prediction[0][0] > 0.009:
        return "Password is Strong!"
    else:
        return "Password is Weak!"


# Get user input
user_input_password = input("Enter a password: ")
result = check_password_strength(user_input_password)
print(result)
