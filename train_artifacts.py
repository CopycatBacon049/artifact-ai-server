import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 1. Load dataset
train_ds = image_dataset_from_directory(
    "artifacts",
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# 2. Base model: MobileNetV2
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # freeze base layers

# 3. Add classifier head
model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Train
model.fit(train_ds, epochs=10)

# 5. Save labels
with open("artifact_labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# 6. Export TensorFlow SavedModel (correct format for FastAPI + Render)
print("Exporting model...")
model.export("ai_model")
print("Model exported!")
