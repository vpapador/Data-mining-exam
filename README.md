# Data-mining-exam

## Cats vs Dogs Image Classification using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) for binary image classification using the Microsoft Cats and Dogs dataset.

The objective is to distinguish between images of cats and dogs while applying the complete deep learning workflow, including data cleaning, preprocessing, data augmentation, model training, and evaluation.

The implementation was developed using TensorFlow and Keras.

## Libraries
Before building the CNN model, the required Python libraries are imported.
TensorFlow is used to create and train the neural network, while additional libraries are used for image preprocessing, dataset management, visualization, and evaluation.

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

## Downloading the Dataset
The dataset is downloaded directly from Microsoft's official source using TensorFlow utilities.

Downloading the dataset programmatically makes the notebook fully reproducible, allowing anyone to execute it without manually downloading the files.

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

## Dataset Organization
The original dataset stores cats and dogs inside separate folders.
The following paths are defined so that the preprocessing stage can easily access every image.

```
base_original = '/root/.keras/datasets/cats_and_dogs_filtered_extracted/PetImages'

cats_dir = os.path.join(base_original, 'Cat')
dogs_dir = os.path.join(base_original, 'Dog')
```

## Dataset Cleaning

The Cats and Dogs dataset contains a number of corrupted image files that cannot be opened correctly.

These files would interrupt the training process, therefore every image is verified using the Python Imaging Library (PIL). Invalid images are automatically discarded before creating the final dataset.

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

## Creating the Dataset Splits
After removing corrupted images, the valid images are randomly divided into three subsets:

70% Training
15% Validation
15% Testing

Using separate datasets allows the model to learn from one subset while being evaluated on unseen data.

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

## Dataset Directories

The newly created folders are assigned to variables that will later be used by TensorFlow's data generators.

Separating the dataset into training, validation and testing directories follows the standard deep learning workflow.


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

## Data Augmentation

To improve the model's ability to generalize, several image transformations are applied during training.

Instead of repeatedly showing the exact same images, random modifications such as rotations, zooming, translations and horizontal flipping generate new image variations on every epoch.

This helps reduce overfitting and improves robustness.

Validation and test images are only normalized because they must represent unseen data.


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

## Data Generators

TensorFlow's ImageDataGenerator is used to automatically load images in batches.

Each image is resized to 150 × 150 pixels, normalized, and converted into tensors that can be processed by the CNN.

The training generator applies data augmentation, whereas the validation and testing generators only perform normalization.


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

## Building the CNN

A custom Convolutional Neural Network is created using four convolutional blocks.

Each convolution layer extracts increasingly complex visual features from the images, while max pooling gradually reduces the spatial dimensions.

Finally, fully connected layers perform the binary classification between cats and dogs.

A Dropout layer is included to reduce overfitting.


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


```

## Model Compilation
The model is compiled using:

Adam Optimizer
Binary Crossentropy Loss
Accuracy Metric

Adam provides efficient gradient optimization, while Binary Crossentropy is the standard loss function for binary classification problems.

```

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



## Callbacks
Two callbacks are used during training.

Early Stopping

Training automatically stops if the validation loss stops improving, preventing unnecessary epochs and reducing overfitting.

Reduce Learning Rate

If the validation loss stagnates, the learning rate is automatically reduced, allowing the optimizer to fine-tune the model more effectively.

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

## Training the CNN
The model is trained using the augmented training dataset while monitoring its performance on the validation dataset.

The callbacks automatically manage the learning process by stopping training when necessary and adjusting the learning rate.


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

## Model Evaluation

After training, the CNN is evaluated using the independent testing dataset.

This provides an unbiased estimate of the model's performance on previously unseen images.

```
test_loss, test_acc = model.evaluate(
    test_generator
)


print(
    "Test accuracy:",
    test_acc
)


```

## Training Curves
Finally, the training history is visualized.

Two graphs are generated:

Accuracy
Loss

Comparing training and validation curves helps determine whether the model learned effectively or suffered from overfitting.


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

## Results

The model successfully learned to classify cats and dogs with high accuracy.

The use of data augmentation, dropout and early stopping helped improve generalization and reduced overfitting during training.

## Conclusion

This project demonstrates the complete implementation of a Convolutional Neural Network for binary image classification.

The workflow includes:

Dataset cleaning
Dataset splitting
Image preprocessing
Data augmentation
CNN architecture
Model training
Performance evaluation

The obtained results indicate that CNNs are highly effective for image classification tasks and provide strong performance even with relatively simple architectures.


## Training duration and computational limitations
The model was initially trained for 10 epochs in order to verify that the complete pipeline was working correctly and to observe the initial learning behavior. Afterwards, the training process was extended to a maximum of 40 epochs to allow the CNN to further improve its performance. However, due to the high computational cost of training, the process required a significant amount of time and caused high CPU utilization. Therefore, the 40-epoch training was manually interrupted before completion. The preliminary results showed that the model was successfully learning useful features, with increasing training accuracy and decreasing loss values.





<img width="999" height="451" alt="image" src="https://github.com/user-attachments/assets/8384f27f-33bb-4fc2-ad63-c7d630cf4cf8" />







| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss | Learning Rate |
|------:|------------------:|--------------:|--------------------:|----------------:|--------------:|
| 1 | 65.62% | 0.6253 | 65.41% | 0.6102 | 0.0010 |
| 2 | 67.49% | 0.6056 | 75.25% | 0.5265 | 0.0010 |
| 3 | 69.49% | 0.5863 | 77.79% | 0.4883 | 0.0010 |
| 4 | 72.79% | 0.5449 | 77.39% | 0.4758 | 0.0010 |
| 5 | 74.19% | 0.5189 | 82.11% | 0.4161 | 0.0010 |
| 6 | 76.36% | 0.4984 | 81.25% | 0.4196 | 0.0010 |
| 7 | 77.60% | 0.4741 | 82.83% | 0.3857 | 0.0010 |
| 8 | 79.35% | 0.4498 | 84.45% | 0.3583 | 0.0010 |
| 9 | 79.88% | 0.4388 | 85.60% | 0.3384 | 0.0010 |
| 10 | 81.18% | 0.4202 | 85.36% | 0.3394 | 0.0010 |











# 🐱🐶 Cats vs Dogs Classification using MobileNetV2

## Project Overview

| Category | Description |
|---|---|
| Project Type | Binary Image Classification |
| Task | Cats vs Dogs Recognition |
| Deep Learning Method | Transfer Learning |
| Base Model | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Dataset | Cats and Dogs Image Dataset |
| Input Image Size | 160 × 160 pixels |
| Classes | 2 (Cat, Dog) |
| Classification Type | Binary Classification |
| Output Layer | Sigmoid Activation |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Initial Learning Rate | 0.0001 |
| Fine Tuning Learning Rate | 0.00001 |
| Batch Size | 32 |
| Initial Epochs | 20 |
| Fine Tuning Epochs | 10 |
| Data Augmentation | Rotation, Zoom, Shift, Shear, Flip |
| Evaluation Metrics | Accuracy, Loss |


## Model Architecture

| Layer | Description |
|---|---|
| Input | 160×160×3 RGB Image |
| Feature Extractor | MobileNetV2 (ImageNet pretrained) |
| Transfer Learning | Frozen pretrained layers |
| Pooling Layer | Global Average Pooling 2D |
| Dense Layer | 128 neurons + ReLU activation |
| Regularization | Dropout (0.5) |
| Output Layer | 1 neuron + Sigmoid activation |


## Training Pipeline

| Step | Process |
|---|---|
| 1 | Load dataset from directories |
| 2 | Apply image preprocessing and augmentation |
| 3 | Create training, validation and test generators |
| 4 | Load pretrained MobileNetV2 model |
| 5 | Freeze convolutional layers |
| 6 | Train classification head |
| 7 | Evaluate on test dataset |
| 8 | Fine tune last 30 MobileNetV2 layers |
| 9 | Re-train with lower learning rate |
| 10 | Perform final evaluation |


## Data Augmentation

| Technique | Value |
|---|---|
| Rotation | 40° |
| Width Shift | 20% |
| Height Shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal Flip | Enabled |
| Fill Mode | Nearest |


## Callbacks

| Callback | Purpose |
|---|---|
| EarlyStopping | Stops training when validation loss stops improving |
| ReduceLROnPlateau | Reduces learning rate when performance plateaus |


## Results

| Metric | Value |
|---|---|
| Test Accuracy | Add your result |
| Test Loss | Add your result |
| Fine Tuned Accuracy | Add your result |
| Fine Tuned Loss | Add your result |


## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network API |
| MobileNetV2 | Feature Extraction |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |


| Metric | Value |
|---|---:|
| Test Accuracy | **86.00%** |
| Test Loss | **0.3313** |

