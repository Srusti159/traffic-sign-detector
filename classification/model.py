import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.optimizers import Adam

def build_traffic_sign_classifier(
    input_shape=(128, 128, 3),
    num_classes=43,
    learning_rate=1e-3,
    dropout_rate=0.3
):
    """
    Builds the MobileNetV2-based traffic sign classification model.
    
    Architecture:
        Input (128x128x3)
           ↓
        MobileNetV2 Base (Pretrained on ImageNet, frozen in Stage 1)
           ↓
        Global Average Pooling 2D (1280-dim feature vector)
           ↓
        Batch Normalization (stabilize activations)
           ↓
        Dense(128, activation='relu')
           ↓
        Dropout(0.3) (regularization against overfitting)
           ↓
        Dense(43, activation='softmax') (class probabilities)
    """
    # 1. Base Pretrained Backbone
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Default to frozen for Stage 1
    base_model.trainable = False

    # 2. Forward pipeline
    inputs = Input(shape=input_shape, name='input_image')
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='batch_norm')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(dropout_rate, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='mobilenetv2_traffic_sign_classifier')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def setup_fine_tuning(model, base_model, unfreeze_last_n=30, fine_tune_lr=1e-4):
    """
    Unfreezes the top N layers of MobileNetV2 base for Stage 2 fine-tuning.
    """
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - unfreeze_last_n

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
