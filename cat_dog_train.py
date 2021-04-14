import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten,Activation,Dropout,Conv2D,MaxPool2D
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.callbacks import ModelCheckpoint




path="DATASETS-20210305T140757Z-001/DATASETS/TRAIN.npy"

dataset=np.load(path,allow_pickle=True)

train_inputs=[]
train_targets=[]

for img,target in dataset[:20000]:
  train_inputs.append(img)
  train_targets.append(target)

train_inputs=np.array(train_inputs)
train_targets=np.array(train_targets)

normalised_train_inputs=train_inputs/255

model = keras.Sequential() #CADM CADM CADM
model.add(Conv2D(32,(3,3),padding="same",input_shape=normalised_train_inputs.shape[1:])) #50x50x3
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) #25x25x32
model.add(Conv2D(64,(3,3),padding="same")) 
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) #12x12x64
model.add(Conv2D(128,(3,3),padding="same")) 
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) #6x6x128
model.add(Conv2D(256,(3,3),padding="same",))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) #3x3x256
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
filepath="Models/best_local.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks_list=[checkpoint]
model.fit(normalised_train_inputs,train_targets,validation_split=0.05,batch_size=750,epochs=30,callbacks=callbacks_list,verbose=1)
