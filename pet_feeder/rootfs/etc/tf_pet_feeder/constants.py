import os

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
X_OFFSET = 185
Y_OFFSET = 0
WIDTH = 245
HEIGHT = 220

CLASSES = ['almost_empty', 'partially_filled', 'completely_filled']
EPOCHS = 15

WORK_DIR = os.path.join(os.sep, "config", os.environ.get('WORK_DIR'))
IMAGE_DIR = os.path.join(os.sep, "config", os.environ.get('IMAGE_DIR'))
OUTPUT_DIR = os.path.join(WORK_DIR, "output")

TRAIN_IMAGES_PATH = os.path.join(OUTPUT_DIR, "train_images.npy")
TRAIN_LABELS_PATH = os.path.join(OUTPUT_DIR, "train_labels.npy")
TEST_IMAGES_PATH = os.path.join(OUTPUT_DIR, "test_images.npy")
TEST_LABELS_PATH = os.path.join(OUTPUT_DIR, "test_labels.npy")

MODEL_PATH = os.path.join(OUTPUT_DIR, "trained_model.h5")
