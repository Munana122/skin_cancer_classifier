import os
import io
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


def preprocess_single_image(image_path):
    """
    Loads and prepares ONE image for prediction. Accepts a file path
    (string), a file-like object (e.g. FastAPI's UploadFile.file), or
    raw bytes. Newer Keras versions only accept a path or io.BytesIO
    directly, so anything else gets wrapped in io.BytesIO first.
    """
    if hasattr(image_path, "read"):
        image_path = io.BytesIO(image_path.read())

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # model expects a batch, so add batch dim of 1
    return img_array
