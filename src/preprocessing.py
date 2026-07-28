import os
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 9 ISIC Classes matching your original setup
CLASS_NAMES = [
    'actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented_benign_keratosis',
    'seborrheic_keratosis', 'squamous_cell_carcinoma', 'vascular_lesion'
]

def load_datasets(train_dir, test_dir):
    """Loads and preprocesses training, validation, and testing datasets."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Returns 3 items matching your notebook cell signature
    return train_ds, val_ds, test_ds