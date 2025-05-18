# Common
import os
import numpy as np
from glob import glob 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow & Keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import GlobalAvgPool2D, Dense, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2
from sklearn.metrics import confusion_matrix, classification_report

# Load Class Names
class_names = sorted(os.listdir('../input/coffee-bean-dataset-resized-224-x-224/train'))
n_classes = len(class_names)
class_dis = [len(glob("../input/coffee-bean-dataset-resized-224-x-224/train/" + name + "/*.png")) for name in class_names]

# CSV Info
data = pd.read_csv('../input/coffee-bean-dataset-resized-224-x-224/Coffee Bean.csv')
print(data.head())

# Image Generators
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, validation_split=0.2)
test_gen = ImageDataGenerator(rescale=1./255)

path = "../input/coffee-bean-dataset-resized-224-x-224/"
train_ds = train_gen.flow_from_directory(path + "train", target_size=(256,256), shuffle=True, batch_size=32, subset="training", class_mode='binary')
valid_ds = train_gen.flow_from_directory(path + "train", target_size=(256,256), shuffle=True, batch_size=32, subset="validation", class_mode='binary')
test_ds = test_gen.flow_from_directory(path + "test", target_size=(256,256), shuffle=False, batch_size=32, class_mode='binary')

# Visualisasi Augmentasi
plt.figure(figsize=(10, 8))
for images, labels in train_ds:
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        plt.title(class_names[int(labels[i])])
        plt.axis('off')
    break
plt.suptitle("Contoh Augmentasi Data", fontsize=16)
plt.tight_layout()
plt.show()

# Inisialisasi Model
name = "ResNet50V2"
base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(256,256,3))
base_model.trainable = False

resnet_50V2 = Sequential([
    base_model, 
    GlobalAvgPool2D(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

resnet_50V2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
cbs = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(name + ".keras", save_best_only=True)
]

# Training
history = resnet_50V2.fit(train_ds, validation_data=valid_ds, epochs=50, callbacks=cbs)

# Plot Akurasi dan Loss
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Load Model Terbaik
model = load_model(name + ".keras")

# Evaluasi dan Confusion Matrix
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels)
    y_pred.extend(np.argmax(preds, axis=1))
    if len(y_true) >= test_ds.samples:
        break

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualisasi Prediksi
plt.figure(figsize=(15,20))
i=1
for images, labels in test_ds:
    id = np.random.randint(len(images))
    image = images[id]
    label = labels[id]
    pred_label = class_names[int(np.argmax(model.predict(image[np.newaxis,...])))]

    plt.subplot(5,4,i)
    plt.imshow(image)
    plt.title(f"Org : {class_names[int(label)]} Pred : {pred_label}")
    plt.axis('off')
    i+=1
    if i>=21: break
plt.tight_layout()
plt.show()
