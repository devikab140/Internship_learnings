import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# STRONG CNN MODEL
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train longer
model.fit(X_train, y_train, epochs=12, batch_size=128,
          validation_data=(X_test, y_test))

# Save model
model.save("mnist_cnn_model.h5")
print(" Strong MNIST model saved")





# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load model
# model = load_model("mnist_cnn_model.h5")

# # Canvas
# canvas = np.zeros((400, 400), dtype=np.uint8)
# drawing = False
# last_x, last_y = -1, -1
# predicted_digit = ""

# # -------------------------------
# # Mouse function
# # -------------------------------
# def draw(event, x, y, flags, param):
#     global drawing, last_x, last_y

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         last_x, last_y = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             cv2.line(canvas, (last_x, last_y), (x, y), 255, 25)
#             last_x, last_y = x, y

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False


# # -------------------------------
# # Preprocess like MNIST
# # -------------------------------
# def preprocess_image(img):
#     # Smooth strokes
#     img = cv2.GaussianBlur(img, (7, 7), 0)

#     # Adaptive threshold (better than fixed 50)
#     thresh = cv2.adaptiveThreshold(
#         img, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )

#     # Find contours
#     contours, _ = cv2.findContours(
#         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     if not contours:
#         return None

#     # Largest contour
#     c = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(c)
#     digit = thresh[y:y+h, x:x+w]

#     # Add padding before resize (IMPORTANT)
#     pad = 10
#     digit = cv2.copyMakeBorder(
#         digit, pad, pad, pad, pad,
#         cv2.BORDER_CONSTANT, value=0
#     )

#     # Resize keeping aspect ratio
#     h, w = digit.shape
#     if h > w:
#         new_h = 20
#         new_w = int(w * 20 / h)
#     else:
#         new_w = 20
#         new_h = int(h * 20 / w)

#     digit = cv2.resize(digit, (new_w, new_h))

#     # Center into 28×28
#     canvas_28 = np.zeros((28, 28), dtype=np.uint8)
#     x_offset = (28 - new_w) // 2
#     y_offset = (28 - new_h) // 2
#     canvas_28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

#     # Invert (MNIST style)
#     canvas_28 = cv2.bitwise_not(canvas_28)

#     # Normalize
#     canvas_28 = canvas_28.astype("float32") / 255.0

#     return canvas_28.reshape(1, 28, 28, 1)


# # -------------------------------
# # Window
# # -------------------------------
# cv2.namedWindow("Draw Digit")
# cv2.setMouseCallback("Draw Digit", draw)

# # -------------------------------
# # Main loop
# # -------------------------------
# while True:
#     display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

#     if predicted_digit != "":
#         cv2.putText(display,
#                     f"Prediction: {predicted_digit}",
#                     (10, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 255, 0),
#                     2)

#     cv2.imshow("Draw Digit", display)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('q'):
#         break

#     elif key == ord('c'):
#         canvas[:] = 0
#         predicted_digit = ""

#     elif key == ord('p'):
#         processed = preprocess_image(canvas)

#         if processed is not None:
#             prediction = model.predict(processed)
#             predicted_digit = np.argmax(prediction)
#             confidence = np.max(prediction)

#             print(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}")
#         else:
#             predicted_digit = "Draw digit"

# cv2.destroyAllWindows()


# #flattening an image using OpenCV
# import cv2
# import numpy as np

# image_path = "C:/Users/HP/Pictures/ppt/flowers.jfif"
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# if img is None:
#    raise ValueError("Image not found or unable to load.")    

# print("Original Image Shape:", img.shape)

# #normalize the image to [0, 1]
# img= img / 255.0

# #flatten the image to 1D array
# flat_img = img.reshape(-1)

# print("Flattened Image Shape:", flat_img.shape)
# print("Image Data (first 10 values):", flat_img[:10])












# #Canny edge detection using OpenCV
# import cv2
# import matplotlib.pyplot as plt

# img = cv2.imread("C:/Users/HP/Pictures/ppt/flowers.jfif")

# #convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #Apply canny edge detection
# edges = cv2.Canny(gray, 100, 200)

# #Display the original and edge-detected images
# plt.imshow(edges, cmap='gray')
# plt.title("Canny Edges")
# plt.axis("off")
# plt.show()








# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # ---------- PATHS ----------
# train_data_dir = 'C:/Users/HP/Downloads/archive (2)/PlantVillageDataset'
# validation_data_dir = 'C:/Users/HP/Downloads/archive (2)/PlantVillageDataset'
# test_data_dir = 'C:/Users/HP/Downloads/archive (2)/PlantVillageDataset'

# IMG_SIZE = (64, 64)
# BATCH_SIZE = 32

# # ---------- DATA GENERATORS ----------
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# val_test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# validation_generator = val_test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# test_generator = val_test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# # ---------- CNN MODEL ----------
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # ---------- TRAIN ----------
# model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=validation_generator
# )

# # ---------- EVALUATE ----------
# loss, acc = model.evaluate(test_generator)
# print(f"Test Accuracy: {acc*100:.2f}%")

# # ---------- SAVE MODEL ----------
# model.save('plant_disease_model.h5')












# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt

# # Load the trained model
# model = load_model('plant_disease_model.h5')


# # Load and preprocess the image
# img_path = "C:/Users/HP/Pictures/ppt/leaf.png"


# img_size = (64, 64)

# img = image.load_img(img_path, target_size=img_size)
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
# img_array /= 255.0  # Rescale pixel values

# # Make prediction
# prediction = model.predict(img_array)
# predicted_class = 'Diseased' if prediction[0][0] > 0.5 else 'Healthy'
# print(f"The leaf is predicted to be: {predicted_class}")
# # Display the image with prediction
# plt.imshow(image.load_img(img_path))
# plt.title(f"Prediction: {predicted_class}")
# plt.axis('off')
# plt.show()


# #shape detection using OpenCV
# import cv2
# import numpy as np
# import math

# # -------------------------------
# # Helper functions
# # -------------------------------
# def distance(p1, p2):
#     return np.linalg.norm(p1 - p2)

# def angle(pt1, pt2, pt3):
#     # angle at pt2
#     v1 = pt1 - pt2
#     v2 = pt3 - pt2
#     cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# # -------------------------------
# # Load image
# # -------------------------------
# image = cv2.imread('C:/Users/HP/Pictures/ppt/shapes.png')
# original = image.copy()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# edges = cv2.Canny(blur, 50, 150)

# contours, _ = cv2.findContours(
#     edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# shape_id = 1

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area < 200:
#         continue

#     epsilon = 0.03 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)

#     vertices = len(approx)
#     shape = "Unknown"

#     # -------------------------------
#     # TRIANGLE
#     # -------------------------------
#     if vertices == 3:
#         shape = "Triangle"

#     # -------------------------------
#     # RECTANGLE / SQUARE / DIAMOND
#     # -------------------------------
#     elif vertices == 4:
#         pts = approx.reshape(4, 2)

#         # side lengths
#         sides = []
#         angles = []

#         for i in range(4):
#             sides.append(distance(pts[i], pts[(i + 1) % 4]))
#             angles.append(
#                 angle(pts[i - 1], pts[i], pts[(i + 1) % 4])
#             )

#         sides = np.array(sides)
#         angles = np.array(angles)

#         side_var = np.std(sides)
#         angle_var = np.std(angles)

#         all_sides_equal = side_var < 10
#         right_angles = np.all((angles > 80) & (angles < 100))

#         if all_sides_equal and right_angles:
#             shape = "Square"
#         elif right_angles:
#             shape = "Rectangle"
#         elif all_sides_equal:
#             shape = "Diamond"
#         else:
#             shape = "Quadrilateral"

#     # -------------------------------
#     # PENTAGON / HEXAGON
#     # -------------------------------
#     elif vertices == 5:
#         shape = "Pentagon"
#     elif vertices == 6:
#         shape = "Hexagon"

#     # -------------------------------
#     # CIRCLE vs STAR
#     # -------------------------------
#     else:
#         perimeter = cv2.arcLength(cnt, True)
#         circularity = 4 * math.pi * area / (perimeter * perimeter)

#         hull = cv2.convexHull(cnt)
#         hull_area = cv2.contourArea(hull)
#         solidity = area / hull_area

#         if circularity > 0.75 and solidity > 0.9:
#             shape = "Circle"
#         else:
#             shape = "Star"

#     # -------------------------------
#     # Display one by one
#     # -------------------------------
#     display = original.copy()
#     cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)

#     cv2.putText(
#         display,
#         f"Shape {shape_id}: {shape}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 0, 255),
#         2
#     )

#     cv2.imshow("Detected Shape", display)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     shape_id += 1



















#     import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# # ---------- DATA PREPARATION ----------
# train_data_dir = "C:/Users/HP/Downloads/iris cv"
# validation_data_dir = "C:/Users/HP/Downloads/iris cv"
# test_data_dir = "C:/Users/HP/Downloads/iris cv"

# IMG_SIZE = (150, 150)
# BATCH_SIZE = 32

# # Data augmentation for training data
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# validation_generator = val_test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# test_generator = val_test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )
# # ---------- CNN MODEL ----------
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2, 2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(3, activation='softmax')  # Assuming 3 classes for iris species
# ])
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# # ---------- TRAIN ----------
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=validation_generator
# )
# # ---------- EVALUATE ----------
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')


# model.save('iris_classification_model.h5')

















# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load the trained model
# model = load_model('iris_classification_model.h5')

# img_path = 'C:/Users/HP/Pictures/ppt/iris versicolor.jfif'  

# img = image.load_img(img_path, target_size=(224, 224))
# img = image.img_to_array(img)
# img = img / 255.0
# img = np.expand_dims(img, axis=0)

# predictions = model.predict(img)
# class_names = ['iris-setosa', 'iris-versicolor', 'iris-virginica']

# predicted_class = class_names[np.argmax(predictions)]
# confidence = np.max(predictions) * 100

# print("Predicted Class:", predicted_class)
# print("Confidence:", round(confidence, 2), "%")





























