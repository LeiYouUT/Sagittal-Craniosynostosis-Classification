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
    optimizer = SGD(lr=0.005, decay=0, momentum=0.0, nesterov=True)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model_1 = get_model()
model_1.load_weights(r'your_path_to_the_pretrained_weights\0-degree-1st-_weights.h5')
folder_path=r"your_path_to_the_test_images\validation-8views-0degree-squ\\"
for folder_name in os.listdir(folder_path):
    #print(folder_name)
    view_account = 8
    label0 = np.zeros(view_account)
    label1 = np.zeros(view_account)
    label2 = np.zeros(view_account)
    label3 = np.zeros(view_account)
    value_final=[]
    j = 0
    for file_name in os.listdir(folder_path+'\\'+folder_name):
        filename=folder_path+'\\'+folder_name+'\\'+file_name
        DIM = 299
        img = image.load_img(filename, target_size=(DIM, DIM))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        preds = model_1.predict(x)
        preds = np.squeeze(preds)
        label0[j]= preds[0]
        label1[j]= preds[1]
        label2[j]= preds[2]
        label3[j]= preds[3]
        j=j+1
    value0=np.max(label0)
    value1=np.max(label1)
    value2=np.max(label2)
    value3=np.max(label3)
    with open("your_path_to_the_results\\result-validation-8view_max-0degree.txt",'a') as f:
            f.write(np.str(folder_name)+'###############\n')
            f.write('Anterior:'+np.str(value0)+'\n')
            f.write('Central:'+np.str(value1)+'\n')
            f.write('Posterior:'+np.str(value2)+'\n')
            f.write('Complex:'+np.str(value3)+'\n')
            f.write('###############\n')
            f.close()
