import cv2
import numpy as np
import tensorflow as tf

# Load your trained model (update the path if needed)
model = tf.keras.models.load_model("terrain_classifier_model.keras")

# Terrain class labels (order must match your dataset)
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Smart drone messages for each terrain
terrain_messages = {
    "AnnualCrop": "Crop field detected. Maintain altitude.",
    "Forest": "Stable zone. Expect calm wind.",
    "HerbaceousVegetation": "Open terrain. Good visibility.",
    "Highway": "Road ahead. Avoid low altitude.",
    "Industrial": "Obstacles likely. Keep safe distance.",
    "Pasture": "Livestock zone. Fly slow.",
    "PermanentCrop": "Orchard detected. Watch tree height.",
    "Residential": "Urban area. Fly high & avoid buildings.",
    "River": "Water body. Maintain stability.",
    "SeaLake": "Large water zone. Ensure GPS lock."
}

# Image size used during training
IMG_SIZE = 128

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

print("📷 Starting webcam... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    # Predict terrain
    predictions = model.predict(input_tensor)
    pred_idx = np.argmax(predictions[0])
    pred_class = class_names[pred_idx]
    message = terrain_messages[pred_class]

    # Display prediction and message on screen
    cv2.putText(frame, f"Terrain: {pred_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, message, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("🛰️ Drone Terrain Classifier", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
