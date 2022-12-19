from tensorflow.keras.layers import *
import tensorflow as tf
import keras.backend as K
import tensorflow.keras as keras
import os
import random
import nibabel as nib
import json
import numpy as np
import gc
import time
#get voxresc model
def get_model_B():
#init layer
    inputs = Input([216, 180, 216, 1], name="input")
    SAconv1 = Conv3D(8, (3,3,3), padding='same')(inputs)
    dp1 = Dropout(.4)(SAconv1)
    SAbn1 = BatchNormalization()(dp1)
    SAconv2 = Conv3D(8, (3,3,3),padding='same')(SAbn1)
    SAbn2 = BatchNormalization()(SAconv2)
    MP1 = MaxPooling3D()(SAbn2)
#
    AAconv1 = Conv3D(16, (3,3,3), padding='same')(MP1)
    AAbn1 = BatchNormalization()(AAconv1)
    dp2 = Dropout(.5)(AAbn1)
    AAconv2 = Conv3D(16, (3,3,3),padding='same')(dp2)
    AAbn2 = BatchNormalization()(AAconv2)
    MP2 = MaxPooling3D()(AAbn2)
#
    BAconv1 = Conv3D(32, (3,3,3), padding='same')(MP2)
    BAbn1 = BatchNormalization()(BAconv1)
    BAconv2 = Conv3D(32, (3,3,3),padding='same')(BAbn1)
    BAbn2 = BatchNormalization()(BAconv2)
    dp3 = Dropout(.6)(BAbn2)
    BAconv3 = Conv3D(32, (3,3,3),padding='same')(dp3)
    BAbn3 = BatchNormalization()(BAconv3)
    MP3 = MaxPooling3D()(BAbn3)
#
    CAconv1 = Conv3D(64, (3,3,3), padding='same')(MP3)
    CAbn1 = BatchNormalization()(CAconv1)
    CAconv2 = Conv3D(64, (3,3,3),padding='same')(CAbn1)
    CAbn2 = BatchNormalization()(CAconv2)
    dp4 = Dropout(.7)(CAbn2)
    CAconv3 = Conv3D(64, (3,3,3),padding='same')(dp4)
    CAbn3 = BatchNormalization()(CAconv3)
    MP4 = MaxPooling3D()(CAbn3)
#
    
    FC1 = Dense(128, activation="softmax")(MP4)
    dp5 = Dropout(.3)(FC1)
    FC2 = Dense(64, activation="softmax")(dp5)
    out1 = Dense(2, activation="softmax")(FC2)
    FL = Flatten()(out1)
    out = Dense(2, activation="softmax")(FL)
#model compiler
    model = keras.Model(inputs=inputs,outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def organizeData(datapath = "FinalData"):
    Scans = [f.path for f in os.scandir(datapath)]
    random.shuffle(Scans)
    start = 0
    end = len(Scans)
    step = 20
    SeperatedScans = []
    for i in range(start, end, step):
        x = i
        SeperatedScans.append(Scans[x:x+step])
    return SeperatedScans

def loadData(subjs):
    tmp = []
    Y = []
    for i in subjs:
        tmp.append(nib.load(i).get_fdata())
        if "HC" in i:
            Y.append([0,1])
        else:
            Y.append([1,0])
    tmp = np.asarray(tmp)
    return(tmp.reshape(len(subjs), 216, 180, 216,1), np.asarray(Y))

def fitModel():
    Scans = organizeData()
    print(len(Scans))
    model = get_model_B()
    model.save("ModelSave_bin1B/modelB")
    

    print(model.summary())
    history = []
    t_end = time.time() + 60 * 60 * 1
    count = 0
    while time.time() < t_end:
        
        for i in range(len(Scans)):
            if count > 0:
                model.load_weights('tmpFolderBin1ModelB/weight'+str(count-1)+'.h5')
            X,Y = loadData(Scans[i])
            print(i)
            history.append(model.fit(
                x=X,
                y=Y,
                batch_size=1,
                epochs=20,
                verbose="auto"
            ).history)
            model.save_weights('tmpFolderBin1ModelB/weight'+str(count)+'.h5')
            print(time.time())
            K.clear_session()
            gc.collect()
            count += 1
        json.dump(history, open("trainingFinal_bin1_modelB.json", 'w'))
        
    model.save_weights('FinalWeightsB_bin1/my_checkpoint')

fitModel()