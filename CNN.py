import numpy as np
import os

import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

# image data Generator
#########################
# data_gen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest')
#
# img = load_img('dataset/train/food/.jpg')
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
#
# i = 0
# for batch in data_gen.flow(x, batch_size=1,
#                            save_to_dir='dataset/train/food', save_prefix='pic', save_format='jpg'):
#     i += 1
#     if i > 20:
#         break
#########################

# set multi_image_data.npy
#########################
# caltech_dir = "dataset/train"
# categories = ['food', 'portrait', 'scenery_city', 'scenery_nature']
# nb_classes = len(categories)
#
# image_w = 100
# image_h = 100
#
# pixels = image_h * image_w * 3
#
# X = []
# Y = []
#
# for idx, cat in enumerate(categories):
#
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1
#
#     image_dir = caltech_dir + "/" + cat
#     files = glob.glob(image_dir + "/*.jpg")
#     print(cat, "파일 길이 : ", len(files))
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.array(img)
#
#         X.append(data)
#         Y.append(label)
#
#         if i % 700 == 0:
#             print(cat, " : ", f)
#
# X = np.array(X)
# Y = np.array(Y)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
# xy = (X_train, X_test, Y_train, Y_test)
# np.save("dataset/multi_image.npy", xy)
#
# print("ok", len(Y))
#########################


#
#########################

categories = ['food', 'portrait', 'scenery_city', 'scenery_nature']  # set categories
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

X_train, X_test, Y_train, Y_test = np.load('dataset/multi_image_data.npy', allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])

nb_classes = len(categories)

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

with K.tf_ops.device('/device:GPU:0'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + '/multi_img_classification.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

model.summary()

history = model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_test, Y_test), callbacks=[checkpoint, early_stopping])
print("정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_val_loss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
