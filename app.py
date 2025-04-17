import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
import random
import folium

# Load your trained model
model = tf.keras.models.load_model("best_terrain_classifier_model.keras")

# Class names and smart messages
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

terrain_messages = {
    "AnnualCrop": "🌾 Crop field detected. Maintain medium altitude.",
    "Forest": "🌲 Stable zone. Expect calm wind.",
    "HerbaceousVegetation": "🌿 Open terrain. Good visibility.",
    "Highway": "🛣️ Road ahead. Avoid low altitude.",
    "Industrial": "🏭 Obstructions likely. Maintain max altitude.",
    "Pasture": "🐄 Livestock zone. Maintain safe distance.",
    "PermanentCrop": "Orchard zone. Monitor tree height.",
    "Residential": "🏘️ Urban zone. Avoid low flyovers.",
    "River": "🌊 Water detected. Be cautious of reflections.",
    "SeaLake": "🌅 Large water body. Ensure GPS lock."
}

def suggest_action(label):
    if label in ['River', 'SeaLake']:
        return "🆙 Ascend"
    elif label == 'Residential':
        return "🛑 Hold position"
    else:
        return "✅ Proceed"

def get_fake_location_and_message():
    lat = round(random.uniform(12.90, 13.10), 6)
    lon = round(random.uniform(77.50, 77.70), 6)
    tweaks = [
        "🌬️ Crosswinds likely near coast.",
        "⛰️ Mountainous region. Maintain stability.",
        "🗺️ Flat zone. GPS signal strong.",
        "🚨 Restricted airspace ahead."
    ]
    tweak = random.choice(tweaks)
    return (lat, lon), tweak

def create_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup="Drone Location 📍").add_to(m)
    m.save("minimap.html")
    return "minimap.html"

def classify_terrain(image):
    # 🔥 Fix: Resize image to model input shape
    image = image.convert("RGB").resize((128, 128))  # Adjust if your model expects other size
    image_array = np.array(image) / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(input_tensor)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    message = terrain_messages[predicted_class]
    action = suggest_action(predicted_class)
    (lat, lon), geo_message = get_fake_location_and_message()
    map_path = create_map(lat, lon)

    final_message = f"""
🛰️ Terrain: {predicted_class}
📣 Message: {message}
📌 Geo Update: {geo_message}
🧭 Location: ({lat}, {lon})
🤖 Suggested Action: {action}
"""

    # Text-to-speech
    tts = gTTS(text=message, lang='en')
    tts.save("alert.mp3")

    return final_message, "alert.mp3", map_path

# Gradio interface (🔥 layout keyword removed!)
interface = gr.Interface(
    fn=classify_terrain,
    inputs=gr.Image(type="pil", label="📷 Upload terrain image"),
    outputs=[
        gr.Textbox(label="🧠 Drone Analysis"),
        gr.Audio(label="🔊 Voice Alert"),
        gr.File(label="🗺️ View Map (HTML)")
    ],
    title="🚁 Smart Drone Terrain Classifier",
    description="Upload a terrain image. Get smart predictions, voice alerts, and GPS-based mini-map."
)

interface.launch()