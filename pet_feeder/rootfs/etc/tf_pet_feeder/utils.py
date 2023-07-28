import base64
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .constants import *

def get_next_filename():
    image_file = None
    if os.path.exists(IMAGE_DIR):
        image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
        image_file = image_files[0] if len(image_files) > 0 else None
    return image_file

def get_image_base64(image_filename):
    if not image_filename:
        return ""

    with open(os.path.join(IMAGE_DIR, image_filename), "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode()

    return image_base64

def create_train_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = []
    labels = []

    for i, class_name in enumerate(CLASSES):
        class_dir = os.path.join(WORK_DIR, class_name)

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)

            if os.path.splitext(image_path)[-1].lower() != ".jpg":
                continue

            image = cv2.imread(image_path)

            height, width = image.shape[:2]
            if height!=IMAGE_HEIGHT or width!=IMAGE_WIDTH:
                image = cv2.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])

            cropped_image = image[Y_OFFSET : Y_OFFSET + HEIGHT, X_OFFSET : X_OFFSET + WIDTH]
            images.append(cropped_image)
            labels.append(i)

    images = np.array(images)
    labels = np.array(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    num_classes = len(CLASSES)
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    np.save(TRAIN_IMAGES_PATH, train_images)
    np.save(TRAIN_LABELS_PATH, train_labels)
    np.save(TEST_IMAGES_PATH, test_images)
    np.save(TEST_LABELS_PATH, test_labels)

    return {
        "train_images_shape": train_images.shape,
        "train_labels_shape": train_labels.shape,
        "test_images_shape": test_images.shape,
        "test_labels_shape:": test_labels.shape
    }

def train_model():
    train_images = np.load(TRAIN_IMAGES_PATH)
    train_labels = np.load(TRAIN_LABELS_PATH)
    test_images = np.load(TEST_IMAGES_PATH)
    test_labels = np.load(TEST_LABELS_PATH)

    num_classes = train_labels.shape[1]
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_images,
        train_labels,
        batch_size=32,
        epochs=EPOCHS,
        validation_data=(test_images, test_labels),
    )

    model.save(MODEL_PATH)
