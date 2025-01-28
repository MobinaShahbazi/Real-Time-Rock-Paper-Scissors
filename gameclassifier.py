import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2
dataset_path = "E:\\term7\\CV\\project\\reyh\\our-dataset"
# dataset_path = "E:\\term7\\CV\\project\\reyh\\merged-dataset"

# Initialize lists for images and labels
train_images = []
valid_images = []
test_images = []
train_labels = []
valid_labels = []
test_labels = []

# Label encoder to convert class names to numerical labels
label_encoder = LabelEncoder()
labels = ['Paper', 'Rock', 'Scissors']
label_encoder.fit(labels)

# Function to load images from a given folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (300, 300))  # Resize to 300x300
                images.append(img)
                labels.append(label)
    return images, labels

# Load train, validation, and test data
train_folder = os.path.join(dataset_path, 'train')
valid_folder = os.path.join(dataset_path, 'valid')
test_folder = os.path.join(dataset_path, 'test')

# Load data for each set
train_images, train_labels = load_images_from_folder(train_folder)
valid_images, valid_labels = load_images_from_folder(valid_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# Convert lists to numpy arrays
train_images = np.array(train_images)
valid_images = np.array(valid_images)
test_images = np.array(test_images)

# Convert labels to numerical values
train_labels = label_encoder.transform(train_labels)
valid_labels = label_encoder.transform(valid_labels)
test_labels = label_encoder.transform(test_labels)
# Build the model
model = keras.Sequential([
    keras.layers.AveragePooling2D(6, 3, input_shape=(300, 300, 3)),  # RGB input
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=30,
    batch_size=32,
    validation_data=(valid_images, valid_labels),
)

# Save the model
model.save('model1.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on test images
predictions = model.predict(test_images)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Display some predictions
num_images = 5  # Display 5 test images
plt.figure(figsize=(10, 10))

for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(test_images[i].reshape(300, 300, 3))  # Ensure 3 channels for RGB
    plt.title(f"Pred: {predicted_labels[i]} | True: {test_labels[i]}")
    plt.axis('off')

plt.show()
