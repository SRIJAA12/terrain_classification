# Smart Drone Terrain Classifier 🚁📉

This project uses deep learning to classify satellite terrain images into meaningful categories such as Forest, River, Residential, etc., and displays a smart message to help autonomous drones make navigation decisions.

---

## 🌍 Demo

You can try the app by uploading a terrain image in the Gradio web interface:
```bash
python app.py
```
Then visit: http://127.0.0.1:7860/

---

## 💡 Features
- Classifies terrain images into 10 categories
- Provides smart flying messages based on terrain
- Real-time webcam prediction (via webcam_predictor.py)
- Simple Gradio web interface for image uploads
- Indoor/Outdoor classification support (optional)

---

## 🌐 Dataset
- Dataset: EuroSAT (RGB)
- Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Source: https://github.com/phelber/eurosat

---

## 👩‍💼 Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/terrain-classifier.git
cd terrain-classifier
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure you have the trained model in the same directory:
```bash
terrain_classifier_model.keras
```

---

## 🔠 Usage

To run the Gradio web app:
```bash
python app.py
```

To test real-time webcam terrain prediction:
```bash
python webcam_predictor.py
```

---

## 🧰 Project Structure
```
.
├── app.py                        # Gradio web app
├── webcam_predictor.py          # Real-time webcam terrain predictor
├── terrain_classifier_model.keras
├── indoor_outdoor_model.keras   # Optional
├── requirements.txt
├── assets/                      # (Optional) Example images, icons
└── README.md
```

---

## 🚀 Future Work
- Add GPS-based suggestions
- Deploy on Hugging Face Spaces
- Add audio alerts
- Improve dataset balance and retrain

---

## 🙏 Acknowledgements
- EuroSAT dataset
- TensorFlow/Keras
- Gradio

---

## ✨ Author
[Srijaa A] - April 2025
