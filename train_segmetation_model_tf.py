"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 22, 2024
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import pickle

def train_segmentation_model(pthDL):

    # Load variables from pickle file
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        sxy, classNames, nm = data['sxy'], data['classNames'], data['nm']

    # Check if a GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU")

    # Define function to save the training progress plot
    def save_training_plot(info, pth_save):
        if info.state == "done":
            plt.figure()
            plt.plot(info.history['loss'], label='train_loss')
            plt.plot(info.history['val_loss'], label='val_loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(pth_save, 'training_process.png'))
            plt.close()

    # Load pre-trained ResNet50 model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(None, None, 3))

    # Define the DeepLabV3 model with ResNet50 backbone
    def deeplabv3_model(num_classes):
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = base_model(inputs)
        x = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    model = deeplabv3_model(num_classes=len(classNames))

    # Define training and validation data directories
    pth_train = os.path.join(pthDL, 'training')
    pth_val = os.path.join(pthDL, 'validation')

    # Define data generators with augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        pth_train,
        target_size=(sxy, sxy),
        batch_size=4,
        class_mode='sparse'
    )

    val_generator = val_datagen.flow_from_directory(
        pth_val,
        target_size=(sxy, sxy),
        batch_size=4,
        class_mode='sparse'
    )

    # Define callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(pthDL, 'model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        ),
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_training_plot(logs, pthDL)
        )
    ]

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy'
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=8,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks_list,
        verbose=1
    )

    # Save the trained model
    model.save(os.path.join(pthDL, 'saved_model'))
