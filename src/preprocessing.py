from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train = datagen.flow_from_directory(
        "data/",
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        subset="training"
    )

    val = datagen.flow_from_directory(
        "data/",
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        subset="validation"
    )

    return train, val