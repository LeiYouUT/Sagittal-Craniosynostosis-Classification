"""
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
"""
import os
import glob
from tensorflow.keras.preprocessing import image
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    #set up fine-tuning layes:
    model=Model(inputs=base_model.input,outputs=preds)
    optimizer = SGD(lr=0.0005, decay=0, momentum=0.0, nesterov=True)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed
 
def mycrossentropy(y_true, y_pred, e=0.1):
    nb_classes=4
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    gamma=2.
    alpha=.25
    loss3= -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return (1-e-0.1)*loss1 + e*loss2 + e*loss3  
def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./log/')
    return [mcp_save, reduce_lr_loss,tensorboard]
if __name__ == "__main__":    
  import glob

  anterior_cso=glob.glob('/grouped-multiview-0degree/0/*.*')
  central_cso=glob.glob('/grouped-multiview-0degree//1/*.*')
  posterior_cso=glob.glob('/grouped-multiview-0degree//2/*.*')
  complex_cso=glob.glob('/grouped-multiview-0degree//3/*.*')
  data = []
  labels = []
  for i in anterior_cso:   
    image1=image.load_img(i, target_size= (299,299))
    image1=np.array(image1)
    data.append(image1)
    labels.append(0)
  for i in central_cso:   
    image1=image.load_img(i, target_size= (299,299))
    image1=np.array(image1)
    data.append(image1)
    labels.append(1)
  for i in posterior_cso:   
    image1=image.load_img(i, target_size= (299,299))
    image1=np.array(image1)
    data.append(image1)
    labels.append(2)
  for i in complex_cso:   
    image1=image.load_img(i, target_size= (299,299))
    image1=np.array(image1)
    data.append(image1)
    labels.append(3)
  data = np.array(data)
  labels = np.array(labels)
  #labels = to_categorical(labels)
  batch_size=32
  gen = ImageDataGenerator(#horizontal_flip = True,
                         #vertical_flip = True,
                         #width_shift_range = 0.4,
                         #height_shift_range = 0.4,
                         #shear_range=0.4,
                         #zoom_range = 0.4,
                         rotation_range = 15
                        )
  print(len(data),len(labels))
  X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                random_state=9,shuffle= True)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=9,shuffle= True) # 0.25 x 0.8 = 0.2  

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  y_val = to_categorical(y_val)
  
  name_weights = "0-degree" + str('0811') + "_weights.h5"
  callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
  print('there are {} training images, {} val images and {} test images'.format(len(X_train),len(X_val),len(X_test)))    
  model = get_model()
  batch_size = 32 
  generator = gen.flow(X_train, y_train, batch_size = batch_size)
  model.fit_generator(
                generator,
                #validation_data=(val_images, val_labels),
                validation_data=(X_test, y_test),
                steps_per_epoch=len(X_train)/batch_size,
                epochs=300,
                shuffle=True,
                verbose=1,
                #validation_data = (val_images, val_labels),
                callbacks = callbacks)
  
  print(model.evaluate(X_val, y_val))

