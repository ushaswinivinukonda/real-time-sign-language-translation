import json
import cv2
import numpy as np
import tensorflow as tf

# Load trained model and class labels
print("Loading model and class labels...")
model = tf.keras.models.load_model('saved_models/asl_model.h5')

with open('saved_models/class_indices.json', 'r') as file:
    # Convert string keys back to integers for dictionary lookup
    class_labels = {int(key): value for key, value in json.load(file).items()}

# Initialize Webcam
webcam_capture = cv2.VideoCapture(0)

if not webcam_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live prediction. Press 'q' to quit.")

while True:
    ret, frame = webcam_capture.read()
    if not ret:
        break

    # Preprocess frame to match model input requirements (350x350, normalized)
    resized_frame = cv2.resize(frame, (350, 350))
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    # Make Prediction
    predictions = model.predict(input_tensor, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])

    # Display prediction label on the video feed
    display_label = f"Sign: {class_labels.get(predicted_index, 'Unknown')} ({confidence_score:.2f})"
    cv2.putText(frame, display_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Gesture Detection', frame)

    # Press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
webcam_capture.release()
cv2.destroyAllWindows()