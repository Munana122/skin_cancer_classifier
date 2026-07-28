import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes=9):
    """Builds a MobileNetV2 model with data augmentation layers for better generalization."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base_model

def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

def train_model(model, train_ds, val_ds, epochs=15, checkpoint_path=None):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    if checkpoint_path:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True))

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return history

def fine_tune_model(model, base_model, train_ds, val_ds, epochs=10, checkpoint_path=None):
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = compile_model(model, learning_rate=1e-5)
    return train_model(model, train_ds, val_ds, epochs=epochs, checkpoint_path=checkpoint_path)

def evaluate_model(model, test_ds):
    results = model.evaluate(test_ds)
    return dict(zip(model.metrics_names, results))

def save_model(model, path):
    model.save(path)