import os
import json
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. Setup Kaggle API Credentials
credentials_file = "kaggle.json"
if os.path.exists(credentials_file):
    with open(credentials_file, "r") as file:
        kaggle_creds = json.load(file)
    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']

import kaggle

# 2. Download and Extract Dataset
dataset_name = "ayuraj/asl-dataset"
zip_filename = "asl-dataset.zip"
extracted_dir = "asl_dataset"

if not os.path.exists(extracted_dir):
    print("Downloading ASL dataset...")
    os.system(f"py -m kaggle datasets download -d {dataset_name}")
    
    with ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall()
    print("Dataset extracted successfully.")

dataset_base_path = os.path.join(extracted_dir, "asl_dataset")
image_size = 350
batch_size = 32

# 3. Data Generators with Validation Split
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_base_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    dataset_base_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Save class indices mapping for inference use
class_mapping = {index: label for label, index in train_generator.class_indices.items()}
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/class_indices.json", "w") as outfile:
    json.dump(class_mapping, outfile)

# 4. Build Model using MobileNetV2 Transfer Learning
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable = False  # Freeze pre-trained layers

features = base_model.output
features = GlobalAveragePooling2D()(features)
features = Dense(128, activation="relu")(features)
output_predictions = Dense(len(train_generator.class_indices), activation="softmax")(features)

model = Model(inputs=base_model.input, outputs=output_predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. Train Model
print("Starting model training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# 6. Save Trained Model
model.save("saved_models/asl_model.h5")
print("Model saved successfully to saved_models/asl_model.h5")