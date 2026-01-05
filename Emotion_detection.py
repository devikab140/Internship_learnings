import os
os.environ["DEEPFACE_LOG_LEVEL"] = "ERROR"

# ===============================================
# 1) IMPORT LIBRARIES
# ===============================================
import os
import cv2
import numpy as np

from mtcnn import MTCNN
from deepface import DeepFace

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ===============================================
# 2) CONFIGURATION
# ===============================================
DATASET_PATH = "emotion_dataset"   # your dataset
IMG_SIZE = (224, 224)              # required for VGG-Face
EPOCHS = 40
BATCH_SIZE = 16


# ===============================================
# 3) INITIALIZE FACE DETECTOR
# ===============================================
detector = MTCNN()


# ===============================================
# 4) FACE → DEEPFACE EMBEDDING FUNCTION
# ===============================================
def get_embedding(face_rgb):
    """
    Extract VGG-Face embedding using DeepFace
    """
    embedding = DeepFace.represent(
        img_path = face_rgb,
        model_name = "VGG-Face",
        enforce_detection = False,
        detector_backend = "skip"
    )
    return np.array(embedding[0]["embedding"])


# ===============================================
# 5) BUILD EMBEDDING DATASET
# ===============================================
X = []
y = []

print(" Extracting embeddings from images...")

for emotion in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion)

    for img_name in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if not faces:
            continue

        x, y0, w, h = faces[0]["box"]
        x, y0 = max(0, x), max(0, y0)

        face = rgb[y0:y0+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, IMG_SIZE)

        embedding = get_embedding(face)

        X.append(embedding)
        y.append(emotion)

print(f" Total samples: {len(X)}")


# ===============================================
# 6) ENCODE LABELS
# ===============================================
X = np.array(X)
EMBEDDING_SIZE = X.shape[1]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)


# ===============================================
# 7) TRAIN / TEST SPLIT
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# ===============================================
# 8) EMOTION CLASSIFIER (SMALL NN)
# ===============================================
model = Sequential([
    Dense(512, activation="relu", input_shape=(EMBEDDING_SIZE,)),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ===============================================
# 9) TRAINING
# ===============================================
print("Training emotion classifier...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save("emotion_deepface_classifier.h5")
print("Model saved")


# ===============================================
# 10) EVALUATION
# ===============================================
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# ===============================================
# 11) LIVE WEBCAM EMOTION DETECTION
# ===============================================
print("\n Starting webcam (press Q to quit)")

model = load_model("emotion_deepface_classifier.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for face_data in faces:
        x, y0, w, h = face_data["box"]
        x, y0 = max(0, x), max(0, y0)

        face = rgb[y0:y0+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, IMG_SIZE)

        embedding = get_embedding(face)
        embedding = np.expand_dims(embedding, axis=0)

        pred = model.predict(embedding, verbose=0)
        emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]

        cv2.rectangle(frame, (x, y0), (x+w, y0+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y0-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("DeepFace Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
