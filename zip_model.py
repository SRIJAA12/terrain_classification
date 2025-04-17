import zipfile

# Path to your model
model_path = "best_terrain_model.keras"
zip_path = "best_terrain_model.zip"

# Create a zip file
with zipfile.ZipFile(zip_path, 'w') as zipf:
    zipf.write(model_path)

print("✅ Model zipped as best_terrain_model.zip")
