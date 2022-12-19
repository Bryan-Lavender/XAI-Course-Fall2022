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
def get_model_C():
#init layer
    inputs = Input([216, 180, 216, 1], name="input")
    SAconv1 = Conv3D(32, (3,3,3), strides = 2, padding='same')(inputs)
    SAbn1 = BatchNormalization()(SAconv1)
    #first additive and normalized
    SAconv2 = Conv3D(32, (3,3,3))(SAbn1)
    SAbn2 = BatchNormalization()(SAconv2)

#first transition conv layer
    Mconv1 = Conv3D(64, (3,3,3), strides = 2, padding='same')(SAbn2)
    Mbn1 = BatchNormalization()(Mconv1)

#first 2-way residual layer
    Aconv1 = Conv3D(64, (3,3,3), padding='same')(Mbn1)
    Abn1 = BatchNormalization()(Aconv1)
    Ac1 = Activation("relu")(Abn1)
    Add1 = Add()([Mbn1, Ac1])
    
    #second additive and normalized
    Bconv1 = Conv3D(64, (3,3,3),padding='same')(Ac1)
    dp1 = Dropout(0.6)(Bconv1)
    Bbn1 = BatchNormalization()(dp1)
    Bc1 = Activation("relu")(Bbn1)
    Bdd1 = Add()([Mbn1, Bc1])
    #second additive and normalized
    
#Second transition conv layer
    Mconv2 = Conv3D(64, (3,3,3), strides = 2, padding='same')(Bdd1)
    Mbn2 = BatchNormalization()(Mconv2)

#Second residual layer
    Aconv2 = Conv3D(64, (3,3,3), padding='same')(Mbn2)
    Abn2 = BatchNormalization()(Aconv2)
    Ac2 = Activation("relu")(Abn2)
    Add2 = Add()([Mbn2, Ac2])

    #Third additive and normalized
    Bconv2 = Conv3D(64, (3,3,3), padding='same')(Add2)
    dp2 = Dropout(.5)(Bconv2)
    Bbn2 = BatchNormalization()(dp2)
    Bc2 = Activation("relu")(Bbn2)
    Bdd2 = Add()([Add2, Bc2])
    
#Third transition conv layer
    Mconv3 = Conv3D(128, (3,3,3), padding='same', strides = 2)(Bdd2)
    Mbn3 = BatchNormalization()(Mconv3)

#Third residual layer
    Aconv3 = Conv3D(128, (3,3,3), padding='same')(Mbn3)
    Abn3 = BatchNormalization()(Aconv3)
    Ac3 = Activation("relu")(Abn3)
    Add3 = Add()([Mbn3, Ac3])
    
    #final additive and normalized
    Bconv3 = Conv3D(128, (3,3,3), padding='same')(Add3)
    dp3 = Dropout(.6)(Bconv3)
    Bbn3 = BatchNormalization()(dp3)
    Bc3 = Activation("relu")(Bbn3)
    Bdd3 = Add()([Add3, Bc3])

#FinalLayers
    MP = MaxPooling3D()(Bdd3)
    dp4 = Dropout(.6)(MP)
    DenseOut = Dense(128, activation="softmax")(dp4)
    Out1 = Dense(2, activation="softmax")(DenseOut)
    fl = Flatten()(Out1)
    Out = Dense(2, activation="softmax")(fl)
#model compiler
    model = keras.Model(inputs=inputs,outputs=Out)
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
    model = get_model_C()
    model.save("ModelSave_bin_small1C/modelC")
    

    print(model.summary())
    history = []
    t_end = time.time() + 60 * 60 * 1
    count = 0
    while time.time() < t_end:
        
        for i in range(len(Scans)):
            if count > 0:
                model.load_weights('tmpFolderBin1ModelC/weight'+str(count-1)+'.h5')
            X,Y = loadData(Scans[i])
            print(i)
            history.append(model.fit(
                x=X,
                y=Y,
                batch_size=1,
                epochs=20,
                verbose="auto"
            ).history)
            model.save_weights('tmpFolderBin1ModelC/weight'+str(count)+'.h5')
            print(time.time())
            K.clear_session()
            gc.collect()
            count += 1
        json.dump(history, open("trainingFinal_bin1_modelC.json", 'w'))
        
    model.save_weights('FinalWeightsC_bin1/my_checkpoint')

fitModel()