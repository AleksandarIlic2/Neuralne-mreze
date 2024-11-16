import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

main_path = './PROJEKAT'

img_size = (80, 80)
batch_size = 64

from keras.utils import image_dataset_from_directory

Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)



classes = Xtrain.class_names

print(classes)

train_class_counts = {"rock": 0, "paper": 0, "scissors": 0}

for images, labels in Xtrain:
    for label in labels.numpy():
        class_name = classes[label]
        train_class_counts[class_name] = train_class_counts.get(class_name) + 1

val_class_counts = {"rock": 0, "paper": 0, "scissors": 0}
for images, labels in Xval:
    for label in labels.numpy():
        class_name = classes[label]
        val_class_counts[class_name] = val_class_counts.get(class_name) + 1

num_images_per_class = [0, 0, 0]
for i in range(3):
    num_images_per_class[i] = val_class_counts[classes[i]] + train_class_counts[classes[i]]

plt.figure()
plt.bar('Xtrain', train_class_counts.get('rock')+train_class_counts.get('paper')+train_class_counts.get('scissors'))
plt.bar('Xval', val_class_counts.get('rock')+val_class_counts.get('paper')+val_class_counts.get('scissors'))
plt.xlabel('Klase')
plt.ylabel('Broj slika')
plt.title('Broj slika po klasama')
plt.show()

plt.figure()
plt.bar(classes, num_images_per_class)
plt.xlabel('Klase')
plt.ylabel('Broj slika')
plt.title('Broj slika po klasama')
plt.show()

plt.figure()
plt.pie(num_images_per_class, labels=classes, autopct='%1.1f%%')
plt.title('Broj slika po klasama')
plt.show()

N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N / 2), i + 1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')

plt.show()

from keras import layers
from keras import Sequential

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical", input_shape=(img_size[0],
                                                                  img_size[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.5),
        layers.RandomContrast(0.25)
    ]
)

plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N / 2), i + 1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')

plt.show()

plt.figure()
counter = 1
printedRock = 0
printedPaper = 0
printedScissors = 0
while 1:
    for img, lab in Xtrain.take(counter):
        for i in range(batch_size):

            if classes[lab[i]] == 'rock' and printedRock == 0:
                plt.subplot(1, 3, 1)
                plt.imshow(img[i].numpy().astype('uint8'))
                plt.title(classes[lab[i]])
                plt.axis('off')
                printedRock = 1

            if classes[lab[i]] == 'scissors' and printedScissors == 0:
                plt.subplot(1, 3, 2)
                plt.imshow(img[i].numpy().astype('uint8'))
                plt.title(classes[lab[i]])
                plt.axis('off')
                printedScissors = 1

            if classes[lab[i]] == 'paper' and printedPaper == 0:
                plt.subplot(1, 3, 3)
                plt.imshow(img[i].numpy().astype('uint8'))
                plt.title(classes[lab[i]])
                plt.axis('off')
                printedPaper = 1
            if printedRock == 1 and printedScissors == 1 and printedPaper == 1:
                break
    counter = counter + 1
    if printedRock == 1 and printedScissors == 1 and printedPaper == 1:
        break



plt.show()

from keras import Sequential
from keras import layers
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy

num_classes = len(classes)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(80, 80, 3)),
    layers.Conv2D(16, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 5, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

from keras.callbacks import EarlyStopping

stop_early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    Xtrain,
    epochs=50,
    validation_data=Xval,
    callbacks=[stop_early],
    verbose=0
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')


labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score

print('Tačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()


labels2 = np.array([])
pred2 = np.array([])
for img, lab in Xval:
    labels2 = np.append(labels, lab)
    pred2 = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

cm2 = confusion_matrix(labels2, pred2, normalize='true')
cmDisplay2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=classes)
cmDisplay2.plot()
plt.show()


goodExamples = []
badExamples = []
counterGood = 0
counterBad = 0

for images,labels in Xval:
    predictions = model.predict(images)
    for i in range(len(images)):
        true_label = classes[labels[i]]
        predLabel = classes[np.argmax(predictions[i])]

        if true_label == predLabel:
            counterGood = counterGood + 1
            if counterGood <= 3:
                goodExamples.append((images[i], true_label, predLabel))
        elif true_label != predLabel:
            counterBad = counterBad + 1
            if counterBad <=3:
                badExamples.append((images[i], true_label, predLabel))

        if counterGood >= 3 and counterBad >= 3:
            break

plt.figure()


for i, (image, true_label, predLabel) in enumerate(goodExamples):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image.numpy().astype('uint8'))
    plt.title(f"True: {true_label}\nPred: {predLabel}")
    plt.axis('off')


for i, (image, trueLabel, predLabel) in enumerate(badExamples):
    plt.subplot(2, 3, 3 + i + 1)
    plt.imshow(image.numpy().astype('uint8'))
    plt.title(f"True: {trueLabel}\nPred: {predLabel}")
    plt.axis('off')

plt.show()
