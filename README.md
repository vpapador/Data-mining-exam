# Data-mining-exam

## Loading the libraries
```
import tensorflow as tf
print(tf.__version__)

import numpy as np
import os
import shutil
from PIL import Image

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

```

## Load the dataset

```
dataset_url = (
    'https://download.microsoft.com/download/3/E/1/'
    '3E1C3F21-ECDB-4869-8368-6DEBA77B919F/'
    'kagglecatsanddogs_5340.zip'
)


dataset_path = tf.keras.utils.get_file(
    'cats_and_dogs_filtered.zip',
    dataset_url,
    extract=True
)


print(os.path.dirname(dataset_path))

```

## Dataset paths

```
base_original = '/root/.keras/datasets/cats_and_dogs_filtered_extracted/PetImages'

cats_dir = os.path.join(base_original, 'Cat')
dogs_dir = os.path.join(base_original, 'Dog')
```

## Clean the dataset

I first openned the zip file on my computer and noticed that each folder contains some broken files
```
output_dir = '/root/.keras/datasets/cats_and_dogs'


if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


folders = [
    'train/cats',
    'train/dogs',
    'validation/cats',
    'validation/dogs',
    'test/cats',
    'test/dogs'
]


for folder in folders:
    os.makedirs(
        os.path.join(output_dir, folder),
        exist_ok=True
    )



def valid_image(path):

    try:
        with Image.open(path) as img:
            img.verify()

        return True

    except:
        return False


```

## Then we split the dataset 

```
def split_and_copy(source_dir, class_name):

    images = []


    for file in os.listdir(source_dir):

        path = os.path.join(source_dir, file)


     
        if valid_image(path):
            images.append(file)

        else:
            print(
                "Removed:",
                file
            )


    print(
        class_name,
        "valid images:",
        len(images)
    )


    # 70% train
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )


    # 15% validation
    # 15% test
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )


    splits = {
        'train': train_imgs,
        'validation': val_imgs,
        'test': test_imgs
    }



    for split, files in splits.items():

        destination = os.path.join(
            output_dir,
            split,
            class_name
        )


        for file in files:

            shutil.copy(
                os.path.join(source_dir,file),
                os.path.join(destination,file)
            )


    print(
        class_name,
        "TRAIN:",
        len(train_imgs),
        "VAL:",
        len(val_imgs),
        "TEST:",
        len(test_imgs)
    )




split_and_copy(
    cats_dir,
    'cats'
)


split_and_copy(
    dogs_dir,
    'dogs'
)

```

## We set the directories

```
base_dir = output_dir


train_dir = os.path.join(
    base_dir,
    'train'
)


validation_dir = os.path.join(
    base_dir,
    'validation'
)


test_dir = os.path.join(
    base_dir,
    'test'
)



print(
    os.listdir(train_dir)
)

print(
    os.listdir(validation_dir)
)

```

## Ready for data augmentation

```

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,

    shear_range=0.2,
    zoom_range=0.2,

    horizontal_flip=True,
    fill_mode='nearest'
)



validation_datagen = ImageDataGenerator(
    rescale=1./255
)



test_datagen = ImageDataGenerator(
    rescale=1./255
)


```

## Generators


```
train_generator = train_datagen.flow_from_directory(
    train_dir,

    target_size=(150,150),

    batch_size=32,

    class_mode='binary'
)



validation_generator = validation_datagen.flow_from_directory(
    validation_dir,

    target_size=(150,150),

    batch_size=32,

    class_mode='binary'
)



test_generator = test_datagen.flow_from_directory(
    test_dir,

    target_size=(150,150),

    batch_size=32,

    class_mode='binary',

    shuffle=False
)



print(
    train_generator.class_indices
)

```

## Building the CNN model

```
def build_cnn():

    model = Sequential([


        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(150,150,3)
        ),

        MaxPooling2D(2,2),



        Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D(2,2),



        Conv2D(
            128,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D(2,2),



        Conv2D(
            128,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D(2,2),



        Flatten(),


        Dense(
            128,
            activation='relu'
        ),


        Dropout(0.5),



        Dense(
            1,
            activation='sigmoid'
        )

    ])


    return model




model = build_cnn()



model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),

    loss='binary_crossentropy',

    metrics=['accuracy']
)



model.summary()


```

## We create call backs

```
early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True

)



lr_reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.3,

    patience=3

)

```

## Training

```

history = model.fit(

    train_generator,

    epochs=40,

    validation_data=validation_generator,

    callbacks=[
        early_stop,
        lr_reduce
    ]

)

```

## Test evaluation

```
test_loss, test_acc = model.evaluate(
    test_generator
)


print(
    "Test accuracy:",
    test_acc
)


```

## Now we create the curves

```
plt.figure(figsize=(12,5))


plt.subplot(1,2,1)

plt.plot(
    history.history['accuracy'],
    label='train'
)

plt.plot(
    history.history['val_accuracy'],
    label='validation'
)

plt.title("Accuracy")

plt.legend()



plt.subplot(1,2,2)

plt.plot(
    history.history['loss'],
    label='train'
)

plt.plot(
    history.history['val_loss'],
    label='validation'
)

plt.title("Loss")

plt.legend()



plt.show()
```
