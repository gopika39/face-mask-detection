# train.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Load dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load pre-trained model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

# Build model
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train, validation_data=val, epochs=5)

# Save model
model.save("model.h5")

print("Model trained and saved!")