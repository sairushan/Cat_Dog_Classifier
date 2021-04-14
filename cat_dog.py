import numpy as np
import cv2
import os


parent="PetImages/"
folders=os.listdir(parent)

dataset=[]

classes=["Cat","Dog"]
for folder in folders:
    full_path=os.path.join(parent,folder)
    target=classes.index(folder)
    imgs=os.listdir(full_path)
    for img in imgs:
        try:
            img_path=os.path.join(full_path,img)
            img=cv2.imread(img_path)
            resized_img=cv2.resize(img,(50,50))
            dataset.append([resized_img,target])
            #cv2.imshow("img",resized_img)
            #cv2.waitKey(1)
        except:
            print("error")
dataset=np.array(dataset)
print(dataset.shape)
np.save("train.npy",dataset)

model=keras.Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=normalised_train_inputs.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',optimizer= keras.optimizers.Adam(learning_rate=0.05),metrics=['accuracy'])


model.fit(normalised_train_inputs,train_targets,batch_size=200,epochs=30)
