import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your rock classifier model
model = load_model('opencv/models/rockclassifier.h5')

# Function to preprocess the input image
def preprocess_image(image):
    # Implement any preprocessing steps needed for your model
    # For example, resizing the image to match the input size of your model
    # You might need to adjust this based on your specific model requirements
    resized_image = cv2.resize(image, (256, 256))
    preprocessed_image = np.expand_dims(resized_image / 255, 0)
    return preprocessed_image

# Function to predict if there is a rock in the image
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    one_hot_prediction = np.eye(prediction.shape[1])[np.argmax(prediction, axis = 1)]
    return one_hot_prediction

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust if needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('Camera Feed', frame)

    # Make predictions on the current frame
    prediction = predict(frame)

    # Output the prediction (customize this based on your model output)
    print('Prediction:', prediction)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
