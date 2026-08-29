import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

def create_data_generators(
    train_dir,
    val_dir,
    img_size=(128, 128),
    batch_size=32
):
    """
    Creates training (with semantic-safe augmentation) and validation generators.
    
    Augmentation reasoning:
    - rotation_range=15: Accounts for vehicle camera tilt or road slope.
    - width/height_shift_range=0.1: Accounts for non-centered bounding box crops.
    - zoom_range=0.15: Simulates variable distance to traffic signs.
    - brightness_range=[0.8, 1.2]: Accounts for weather/shadow changes.
    - horizontal_flip=False: CRITICAL. Prevents altering sign directional meaning (e.g. Left Turn vs Right Turn).
    - rescale=1./255: Normalizes pixel values to [0, 1].
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
        fill_mode='nearest'
    )

    # Validation generator must have NO augmentation and shuffle=False
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator

def preprocess_single_image(img_input, target_size=(128, 128)):
    """
    Applies identical preprocessing to an input image (PIL Image or file path) for inference.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).convert('RGB')
    else:
        img = img_input.convert('RGB')

    img_resized = img.resize(target_size)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_arr, axis=0)
