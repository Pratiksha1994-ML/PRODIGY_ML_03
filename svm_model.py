import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# DATASET SETTINGS
# ==========================================

DATASET_PATH = "PetImages"

CATEGORIES = ["Cat", "Dog"]

IMG_SIZE = 32

data = []

print("Loading dataset...")

# ==========================================
# LOAD IMAGES + EXTRACT HOG FEATURES
# ==========================================

for category in CATEGORIES:

    path = os.path.join(DATASET_PATH, category)

    label = CATEGORIES.index(category)

    print(f"\nProcessing {category} images...")

    # Limit images for faster training
    for img in os.listdir(path)[:5000]:

        try:

            img_path = os.path.join(path, img)

            # Read image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Skip corrupted images
            if image is None:
                continue

            # Resize image
            resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            # ==========================================
            # HOG FEATURE EXTRACTION
            # ==========================================

            features = hog(
                resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            data.append([features, label])

        except Exception:
            pass

print("\nDataset Loaded Successfully!")

# ==========================================
# SHUFFLE DATA
# ==========================================

np.random.shuffle(data)

# ==========================================
# SPLIT FEATURES AND LABELS
# ==========================================

X = []
y = []

for features, label in data:

    X.append(features)

    y.append(label)

X = np.array(X)

y = np.array(y)

print(f"\nTotal Images Used: {len(X)}")

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Linear SVM Model...")

# ==========================================
# TRAIN MODEL
# ==========================================

model = LinearSVC(max_iter=10000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/cat_dog_svm.pkl")

print("\nModel Saved Successfully!")

# ==========================================
# TEST IMAGE PREDICTION
# ==========================================

test_image = "test.jpg"

print("\nPredicting Test Image...")

# Read original image for display
original_img = cv2.imread(test_image)

# Read grayscale image
img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)

if img is None:

    print("Test image not found!")

else:

    # Resize image
    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Extract HOG features
    test_features = hog(
        resized_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # Reshape for prediction
    test_features = test_features.reshape(1, -1)

    # Predict
    prediction = model.predict(test_features)

    # Label
    if prediction[0] == 0:
        label = "CAT 🐱"
    else:
        label = "DOG 🐶"

    print(f"\nPrediction: {label}")

    # ==========================================
    # DISPLAY IMAGE
    # ==========================================

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))

    plt.imshow(original_img)

    plt.title(f"Predicted: {label}", fontsize=16)

    plt.axis("off")

    plt.show()