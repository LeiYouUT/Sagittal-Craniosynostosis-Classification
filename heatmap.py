import os
import glob
from tensorflow.keras.preprocessing import image
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
#import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import  Dropout,Input, Flatten, Dense, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate,LeakyReLU
from tensorflow.keras.utils import to_categorical
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras.utils import to_categorical
from matplotlib.pyplot import imshow

def get_model():
    NB_IV3_LAYERS_TO_FREEZE = 172
    base_model=InceptionV3(weights='imagenet',include_top=False,input_tensor=Input(shape=(299, 299, 3)))
    base_model.trainable=False
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x=Dense(256,kernel_regularizer=tf.keras.regularizers.l2(0.001),activation='relu')(x) #dense layer 3
    x = Dropout(0.3)(x)
    preds=Dense(4,activation='softmax')(x)  
    model=Model(inputs=base_model.input,outputs=preds)
    optimizer = SGD(lr=0.0005, decay=0, momentum=0.0, nesterov=True)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model_1 = get_model()
#print(model_1.summary())
model_1.load_weights(r'your_path_to_the_weight\0-degree0811_weights.h5')


# read the image for heatmap
ORIGINAL = r'.\cropped-new-case-60_0degree_1.png'
DIM = 299
img = image.load_img(ORIGINAL, target_size=(DIM, DIM))
imshow(img)
with tf.GradientTape() as tape:
    last_conv_layer = model_1.get_layer('conv2d_93')
    iterate = tf.keras.models.Model([model_1.inputs], [model_1.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(x)
    class_out = model_out[:, np.argmax(model_out[0])]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((8, 8))
img = cv2.imread(ORIGINAL)
# Old fashioned way to overlay a transparent heatmap onto original image, the same as above
heatmapY = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmapY = cv2.applyColorMap(np.uint8(255*heatmapY), cv2.COLORMAP_JET)  # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
imageY = cv2.addWeighted(heatmapY, 0.5, img, 1.0, 0)
print(heatmapY.shape, img.shape)
# draw the orignal x-ray, the heatmap, and the overlay together
output = np.hstack([img, heatmapY, imageY])
fig, ax = plt.subplots(figsize=(20, 18))
ax.imshow(np.random.rand(1, 99), interpolation='nearest')
plt.imshow(output)
plt.show()
