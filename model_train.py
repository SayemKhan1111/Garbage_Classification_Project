from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os




# Prepare data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "data",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val = datagen.flow_from_directory(
    "data",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build model
model = Sequential([
    MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet'),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

# Compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=5)

# Save model

os.makedirs("models", exist_ok=True)
model.save("models/my_model.keras")