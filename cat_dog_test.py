from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

path="DATASETS-20210305T140757Z-001/DATASETS/TRAIN.npy"

dataset=np.load(path,allow_pickle=True)
classes=["cat","Dog"]
test_inputs=[]
test_targets=[]

model=keras.models.load_model("Models/best.hdf5")
for img,tgt in dataset[20000:]:
    test_inputs.append(img)
    test_targets.append(tgt)

test_inputs=np.array(test_inputs)
test_targets=np.array(test_targets)
normalised_test_inputs=test_inputs/255
normalised_test_inputs1=normalised_test_inputs.reshape(-1,1,50,50,3)
##for i,test in enumerate(normalised_test_inputs1):
##    prediction=model.predict_classes(test)
##    cv2.imshow("image",normalised_test_inputs[i])
##    #cv2.waitKey(0)
##    print("target:",classes[test_targets[i]],"predcition:",classes[prediction[0][0]])
##    if cv2.waitKey(0)==27:
##        break
##
##cv2.destroyAllWindows()
#prediction=model.predict_classes(normalised_test_inputs[0].reshape(1,50,50,3))
model.evaluate(normalised_test_inputs,test_targets)
