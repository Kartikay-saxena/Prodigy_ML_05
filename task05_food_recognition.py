# ==============================
# Task 05 - Food Recognition (Fast Version, 10 Classes)
# ==============================

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ==============================
# Parameters
# ==============================
DATASET_DIR = r"dataset_food_task5/food-101/images/"  # keep same as your code
IMG_SIZE = 32
MAX_IMAGES = 50          # max images per class
NUM_CLASSES = 10         # reduce to 10 classes for demo purposes

data = []
labels = []

# ==============================
# List all food classes (folders)
# ==============================
classes = [c for c in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, c))]
classes = classes[:NUM_CLASSES]  # take first 10 classes only

# ==============================
# Load images
# ==============================
for idx, category in enumerate(classes):
    folder_path = os.path.join(DATASET_DIR, category)
    count = 0
    for file in os.listdir(folder_path):
        if count >= MAX_IMAGES:
            break
        # Skip non-image files
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize and flatten
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img.flatten())
        labels.append(idx)
        count += 1

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)
print("Dataset shape:", X.shape, y.shape)  # e.g. (500, 1024) (500,)

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Train SVM
# ==============================
print("Training SVM model...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# ==============================
# Evaluate model
# ==============================
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Optional: show a few predictions
# ==============================
import matplotlib.pyplot as plt

def show_image(img_array, true_label, pred_label):
    plt.imshow(img_array.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"True: {classes[true_label]} | Pred: {classes[pred_label]}")
    plt.axis("off")
    plt.show()

for i in range(5):
    show_image(X_test[i], y_test[i], y_pred[i])
