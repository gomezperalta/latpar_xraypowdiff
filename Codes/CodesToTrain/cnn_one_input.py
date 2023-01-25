#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:30:00 2021

@author: iG
"""

import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt

import keras.utils as kutils
import keras.models as Models
import keras.layers as Layers
import keras.regularizers as Reg
import keras.optimizers as Optimizers
import keras.initializers as Initializers
import keras.callbacks as callbacks
import keras.losses
from keras import backend as K

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.utils import shuffle

import copy
import time
import os

np.random.seed(3564)
initializer_seed = 10


def ConvLayer(x, filters = 32, filter_size=4, padding = 'same', 
              kernel_initializer='glorot_normal',
              activation='', dropout=0.0,
              stage = 1):
    
    
    if kernel_initializer == 'glorot_normal':
        kernel_initializer = Initializers.glorot_normal(seed = initializer_seed)
        
    elif kernel_initializer == 'glorot_uniform':
        kernel_initializer = Initializers.glorot_uniform(seed = initializer_seed)
    
    elif kernel_initializer == 'he_normal':
        kernel_initializer = Initializers.he_normal(seed = initializer_seed)
        
    elif kernel_initializer == 'he_uniform':
        kernel_initializer = Initializers.he_uniform(seed = initializer_seed)
        
    elif kernel_initializer == 'random_normal':
        kernel_initializer = Initializers.random_normal(seed = initializer_seed)
        
    elif kernel_initializer == 'random_uniform':
        kernel_initializer = Initializers.random_uniform(seed = initializer_seed)
        
    stage = str(stage).zfill(2)
    x = Layers.Conv1D(filters = filters, kernel_size = filter_size, 
                      padding = padding, kernel_initializer=kernel_initializer,
                      bias_initializer="zeros",
                      name = 'CONV1D_' + stage)(x)
    x = Layers.BatchNormalization(name = 'BN_' + stage)(x)
    
    if activation:
        if activation == 'tanh': 
            x = Layers.Lambda(lambda x: (2/3)*x)(x)
            x = Layers.Activation(activation, name = activation + '_' + stage)(x)
            x = Layers.Lambda(lambda x: (7/4)*x)(x)

        else:
            x = Layers.Activation(activation, name = activation + '_' + stage)(x)
            #x = Layers.PReLU(alpha_regularizer=Reg.l1(5e-4))(x)
    if dropout:
        x = Layers.Dropout(dropout)(x)
     
    return x, int(stage) + 1

def ResBlock(x, filters = 32, fsize_main = 4, fsize_sc = 1,
             padding = 'same', kernel_initializer='0.001',
             activation='', dropout=0.0,
             stage = 1, chain = 2):
    
    x_shortcut = x
    layers = stage
    
    if kernel_initializer == 'glorot_normal':
        kernel_initializer = Initializers.glorot_normal(seed = initializer_seed)
        
    elif kernel_initializer == 'glorot_uniform':
        kernel_initializer = Initializers.glorot_uniform(seed = initializer_seed)
    
    elif kernel_initializer == 'he_normal':
        kernel_initializer = Initializers.he_normal(seed = initializer_seed)
        
    elif kernel_initializer == 'he_uniform':
        kernel_initializer = Initializers.he_uniform(seed = initializer_seed)
        
    elif kernel_initializer == 'random_normal':
        kernel_initializer = Initializers.random_normal(seed = initializer_seed)
        
    elif kernel_initializer == 'random_uniform':
        kernel_initializer = Initializers.random_uniform(seed = initializer_seed)
     
    else:
        kernel_initializer = Initializers.VarianceScaling(float(kernel_initializer),
                                                        mode = "fan_avg", distribution="uniform",
                                                        seed = initializer_seed)

    for depth in range(chain):

        if depth%2 == 0:
            x, layers = ConvLayer(x, filters = filters, filter_size = fsize_main, 
                                  kernel_initializer = kernel_initializer,
                                  activation = activation, stage = layers, dropout=dropout)
        else:
            x, layers = ConvLayer(x, filters = filters, filter_size = fsize_main,
                                  kernel_initializer = kernel_initializer,
                                  dropout=dropout, stage=layers)

    if K.int_shape(x_shortcut)[-1] != K.int_shape(x)[-1]:
        x_shortcut, layers = ConvLayer(x_shortcut, filters = filters, filter_size = fsize_sc,
                                       kernel_initializer = kernel_initializer,
                                       dropout=dropout, stage=layers)
    layers -= 1
    x = Layers.Add(name='addition_' + str(layers).zfill(2))([x, x_shortcut])
    
    if activation:
        if activation == 'tanh': 
            x = Layers.Lambda(lambda x: (2/3)*x)(x)
            x = Layers.Activation(activation)(x)
            x = Layers.Lambda(lambda x: (7/4)*x)(x)

        else:
            x = Layers.Activation(activation)(x)

#    if dropout:
#        x = Layers.Dropout(dropout)(x)
     
    return x, layers + 1

def ctrl_dictionary(archivo='model_control_file'):
    """
    Funcion copiada de patolli
    """
    
    f=list(filter(None,open(str(archivo)+'.txt','r').read().split('\n')))

    sg_ikeys=[f.index(sg) for sg in f if 'NAME' in sg]+[len(f)]
    
    diccio={}
    for item in range(len(sg_ikeys)-1):
        text = f[sg_ikeys[item]:sg_ikeys[item+1]]
        key = [entry.split(':')[0] for entry in text]
        value = [entry.split(':')[1].strip() for entry in text]
        diccio[item] = {k:v for k,v in zip(key,value)}
        
    return diccio

def model(input_shape = (1,1), output_dims = 1, conv_arch = list(), conv_act = list(),
           conv_filters = list(), conv_dropout = [0.0], res_act = list(),
           res_filters = list(), res_chain =2,
           fc_dropout = 0.0, fc_act = 'relu',
           filter_size = [8], filter_sc_size = [1], pool_size = 4,
           pool_stride = 1, hl_frac = list(), beta_1 = 0.9,
           beta_2 = 0.999, decay = 1e-6, lr = 1e-3, task = 'regression'):

    convs = conv_arch.count('Conv')
    resblocks = conv_arch.count('ResBlock')
    pools = conv_arch.count('Pool')

    if len(conv_act) == 1:
        conv_act = [conv_act[0],]*convs

    if len(res_act) == 1:
        res_act = [res_act[0],]*resblocks

    if len(filter_size) == 1:
    	filter_size = [filter_size[0],]*resblocks
    
    if len(filter_sc_size) == 1:
    	filter_sc_size = [filter_sc_size[0],]*resblocks

    if len(conv_dropout) == 1:
        conv_dropout = [conv_dropout[0],]*resblocks

    input_layer = Layers.Input(shape = input_shape, name = 'input_layer')
    
    conv_count = 0
    res_count = 0
    pool_count = 0
    
    fn = 0
    fn_sc = 0
    conv_n = 0
    
    for item, layer in enumerate(conv_arch):
        
        if item == 0:
            
            if layer == 'Conv':
                x, layers = ConvLayer(input_layer, conv_filters[conv_count],
                              filter_size = filter_size, 
                              activation = conv_act[conv_count],
                              dropout = conv_dropout,
                              stage = item+1)
                
                conv_count += 1
                
            elif layer == 'ResBlock':
                x, layers = ResBlock(input_layer, res_filters[res_count] , 
                         fsize_main = filter_size[fn],
                         fsize_sc = filter_sc_size[fn_sc], 
                         activation = res_act[res_count],
                         dropout = conv_dropout[conv_n],
                         stage = item+1, chain=res_chain)
                res_count += 1
                fn += 1
                fn_sc += 1
                conv_n += 1
        else:
            
            if layer == 'Conv':
                x, layers = ConvLayer(x, conv_filters[conv_count],
                              filter_size = filter_size[fn], 
                              activation = conv_act[conv_count],
                              dropout = conv_dropout[conv_n],
                              stage = layers)
                conv_count += 1
                fn += 1
                fn_sc += 1
                conv_n += 1
            
            elif layer == 'Pool':
                #x = Layers.MaxPool1D(pool_size=pool_size[pool_count], 
                x = Layers.AveragePooling1D(pool_size=pool_size[pool_count], 
                                     stride=pool_stride[pool_count],
                                     name = 'POOL_' + str(layers).zfill(2))(x)
                layers += 1
                pool_count += 1

            elif layer == 'Permute':
                x = Layers.Permute((2,1))(x)

            elif layer == 'ResBlock':
                x, layers = ResBlock(x, res_filters[res_count] , 
                                     fsize_main = filter_size[fn],
                                     fsize_sc = filter_sc_size[fn_sc], 
                                     activation = res_act[res_count],
                                     dropout = conv_dropout[conv_n],
                                     stage = layers, chain=res_chain)
                res_count += 1
                fn += 1
                fn_sc += 1
                conv_n += 1


    x = Layers.GlobalAveragePooling1D()(x)
    x = Layers.BatchNormalization(name = 'BN_Flatten')(x)

    kinit = Initializers.VarianceScaling(float(0.001), mode = "fan_avg", distribution="uniform", seed=initializer_seed)
    
    hlnum = 2
    hidden_layers = [int(K.int_shape(x)[-1]*hl) for hl in hl_frac]
    
    for nhl, hl in enumerate(hidden_layers):
        x = Layers.Dense(hl, name = 'FC' + str(hlnum), kernel_initializer=kinit)(x)

        if fc_act == 'tanh': 
            x = Layers.Lambda(lambda x: (2/3)*x)(x)
            x = Layers.Activation(fc_act, name = fc_act + '_' + str(hlnum))(x)
            x = Layers.Lambda(lambda x: (7/4)*x)(x)

        else:
            x = Layers.Activation(fc_act, name = fc_act + '_' + str(hlnum))(x)
        hlnum += 1
        
        if fc_dropout:
            x = Layers.Dropout(fc_dropout)(x)
    
    if task == 'classification' and output_dims == 1:
        x = Layers.Dense(output_dims, name='output_layer', kernel_initializer=kinit)(x)    
        output_layer = Layers.Activation('sigmoid', name='sigmoid_ouput')(x)
    
    elif task == 'classification' and output_dims != 1:
        x = Layers.Dense(output_dims, name='output_layer', kernel_initializer=kinit)(x)    
        output_layer = Layers.Activation('softmax', name='softmax_ouput')(x)
    
    else:
        output_layer = Layers.Dense(output_dims, name='output_layer', kernel_initializer=kinit)(x)    
    
    
    modelo = Models.Model(inputs=input_layer, outputs=output_layer)
    
    if task == 'classification' and output_dims == 1:
        modelo.compile(loss='binary_crossentropy', 
                      optimizer=Optimizers.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                      metrics=['accuracy'])
    elif task == 'classification' and output_dims != 1:
        modelo.compile(loss='categorical_crossentropy', 
                      optimizer=Optimizers.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                      metrics=['accuracy'])
    else:
        modelo.compile(loss='logcosh',
                      optimizer=Optimizers.Adamax(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,))
    
    return modelo

def training(model, X, Y, epochs=300, xval=np.zeros((1,1,1)), yval=np.zeros((1,1,1)), batch_size=16, saveas='modelo_nn', validation_freq=2,
             verbose=1, task='regression'):
    
    """
    Funcion copiada de patolli
    """
    modelCheckpoint=callbacks.ModelCheckpoint(str(saveas)+'.h5', monitor='val_loss', 
                                                    verbose=0, save_best_only=True, mode='auto')
    history = callbacks.History()
    data = model.fit(X,Y, validation_data=(xval,yval), epochs=epochs,batch_size=batch_size,
                     callbacks=[modelCheckpoint,history],shuffle=True, verbose=verbose)
    
    try:
        kutils.plot_model(model,to_file=str(saveas)+'.png', show_shapes=True, show_layer_names=True)
    except:
        print('It was not possible to plot the model. GraphViz or Pydot not installed')
    
        
    """ Creacion del archivo csv """

    loss_log = data.history['loss']
    val_loss_log = data.history['val_loss']
    loss_log = np.array(loss_log)
    val_loss_log = np.array(val_loss_log)
    
    if task == 'classification':
        acc_log = data.history['acc']
        val_acc_log = data.history['val_acc']
        acc_log = np.array(acc_log)
        val_acc_log = np.array(val_acc_log)
        
        mat = np.vstack((loss_log, acc_log, val_loss_log, val_acc_log))
    else: 
        mat = np.vstack((loss_log, val_loss_log))
    
    mat = np.transpose(mat)
    dataframe1 = pd.DataFrame(data=mat)
    dataframe1.to_csv(str(saveas)+'.csv', sep=',', header=False, float_format='%.7f', index=False)
    
    return data, dataframe1, model

def split_collection(x = np.zeros((1,1,1)), y = np.zeros((1,1)), df = pd.DataFrame(),
                     test_val_frac=0.15):
    
    if test_val_frac != 0:
        idxtest = np.random.choice(df.index, size = int(test_val_frac*df.shape[0]), replace=False)
        idxtraval = [i for i in range(df.shape[0]) if i not in idxtest]
    
        xtraval = x[idxtraval]
        xtest = x[idxtest]
        
        ytraval = y[idxtraval]
        ytest = y[idxtest]
        
        dftraval = df.take(idxtraval).reset_index(drop=True)
        dftest = df.take(idxtest).reset_index(drop=True)
        
        #np.save('xtraval', xtraval)
        #np.save('xtest', xtest)
        
        np.save('ytraval', ytraval)
        np.save('ytest', ytest)
        
        dftraval.to_csv('dftraval.csv', index=None)
        dftest.to_csv('dftest.csv', index=None)
        
        return xtraval, xtest, dftraval, dftest, ytraval, ytest
    
    else:
        return x, None, df, None, y, None
        

def main_function(control_file = 'txt-file_name',
                  output_dims = 1, test_val_frac=0.15):
    
    #Joining dataframe

    df = pd.read_csv('omdb_hr/dbset_macro.csv') #This is a support file to take interplanar distances
    minpeaks = 6   
    peaks = pd.read_csv('omdb_hr/db_peaks.csv') #File containing the number of peaks with an intensity >= 5% with 50 nm
    cond0 = peaks[peaks['numpeaks'] >= minpeaks].index  #Indices that satisfy condition of minpeaks.    
 
    y = np.load('omdb_hr/yts2_sxv.npy') #This is first called to get the indices of the samples to work with.
    y = y[cond0]
    cond = np.argwhere((y[:,0] <= 30)*(y[:,-1] >= 1.5406))[:,0] #Compounds with interplanar distances within 60 and 1.5406 angstroms
    y = y[cond]

    df = df.take(cond0).reset_index(drop=True)
    df = df.take(cond).reset_index(drop=True) 

    df_temp = pd.read_csv('omdb_hr/dbset_0050.csv')
    df_temp = df_temp.take(cond0).reset_index(drop=True) 
    df_temp = df_temp.take(cond).reset_index(drop=True) 
    df = pd.concat((df,df_temp), ignore_index=True)

    df_temp = pd.read_csv('omdb_hr/dbset_0100.csv')
    df_temp = df_temp.take(cond0).reset_index(drop=True)
    df_temp = df_temp.take(cond).reset_index(drop=True) 
    df = pd.concat((df,df_temp), ignore_index=True)

    df_temp = pd.read_csv('omdb_hr/dbset_0250.csv')
    df_temp = df_temp.take(cond0).reset_index(drop=True)
    df_temp = df_temp.take(cond).reset_index(drop=True) 
    df = pd.concat((df,df_temp), ignore_index=True)
    
    #Joining input_data
    x = np.load('omdb_hr/xset_macro.npy')
    x = x[cond0,:,:] 
    x = x[cond,:,:] 

    x_temp = np.load('omdb_hr/xset_0050.npy')
    x_temp = x_temp[cond0,:,:] 
    x_temp = x_temp[cond,:,:] 
    x = np.concatenate((x,x_temp),axis=0)

    x_temp = np.load('omdb_hr/xset_0100.npy')
    x_temp = x_temp[cond0,:,:]
    x_temp = x_temp[cond,:,:] 
    x = np.concatenate((x,x_temp),axis=0)

    x_temp = np.load('omdb_hr/xset_0250.npy')
    x_temp = x_temp[cond0,:,:]
    x_temp = x_temp[cond,:,:]
    x = np.concatenate((x,x_temp),axis=0)
    x_temp = ''

    #Joining output data
    y = np.concatenate((y,y,y,y))
    
    input_shape = x.shape[1:]
    output_dims = output_dims
    
    idxtest = np.random.choice(df_temp.index, size = int(test_val_frac*df_temp.shape[0]), replace=False)
    idxtest = np.concatenate((idxtest, idxtest + df_temp.shape[0], 
    							idxtest + 2*df_temp.shape[0], idxtest + 3*df_temp.shape[0]))
    
    idxtraval = [i for i in range(df.shape[0]) if i not in idxtest]

    xtraval = x[idxtraval]
    xtest = x[idxtest]
    
    ytraval = y[idxtraval]
    ytest = y[idxtest]
    
    dftraval = df.take(idxtraval).reset_index(drop=True)
    dftest = df.take(idxtest).reset_index(drop=True)
    
    #np.save('xtraval', xtraval)
    #np.save('xtest', xtest)
    np.save('ytraval', ytraval)
    np.save('ytest', ytest)
    
    dftraval.to_csv('dftraval.csv', index=None)
    dftest.to_csv('dftest.csv', index=None)
    df_temp = ''

    #Division del conjunto principal
#    xtraval, xtest, dftraval, dftest, ytraval, ytest = split_collection(x = x, y = y, df = df, test_val_frac=test_val_frac)    
    cnn_diccio = ctrl_dictionary(control_file)
    
    directorio = time.ctime().replace(' ', ':').split(':')[1:]
    directorio.pop(-2)
    directorio = '_'.join(directorio)
    print('directorio', directorio)    
    os.system('mkdir ' + directorio)

    os.system('mv *traval* ' + directorio)
    os.system('mv *test* ' + directorio)
    os.system('cp ' + control_file + '.txt ' + directorio)

    xtraval_or = copy.deepcopy(xtraval)
    ytraval_or = copy.deepcopy(ytraval)
    xtraval, ytraval = shuffle(xtraval,ytraval, random_state=10)
    xtraval, ytraval = shuffle(xtraval,ytraval, random_state=10)
    xtraval, ytraval = shuffle(xtraval,ytraval, random_state=10)    

    for cnn in list(cnn_diccio):
        print('Training ', cnn + 1,'/',len(cnn_diccio.keys()))
        hyperparameters = cnn_diccio[cnn]
        
        model_name = hyperparameters['NAME']    
        conv_arch = [layer.strip() for layer in hyperparameters['CONV_ARCH'].split(',')]
        conv_act = [layer.strip() for layer in hyperparameters['CONV_ACTIVATION'].split(',')]
        conv_filters = [int(layer) for layer in hyperparameters['CONV_FILTERS'].split(',')]
        conv_dropout = [round(float(layer),5) for layer in hyperparameters['DROPOUT_CONV'].split(',')]
        res_act = [layer.strip() for layer in hyperparameters['RES_ACTIVATION'].split(',')]
        res_filters = [int(layer) for layer in hyperparameters['RES_FILTERS'].split(',')]
        res_chain = int(hyperparameters['RES_CHAIN'])
        fc_dropout = round(float(hyperparameters['DROPOUT_FC']),5)
        fc_act = hyperparameters['FC_ACTIVATION']
        filter_size = [int(layer) for layer in hyperparameters['FILTER_SIZE'].split(',')]
        filter_sc_size = [int(layer) for layer in hyperparameters['FILTER_SC_SIZE'].split(',')]
        pool_size = [int(layer) for layer in hyperparameters['POOL_SIZE'].split(',')]
        pool_stride = [int(layer) for layer in hyperparameters['POOL_STRIDE'].split(',')]
        hl_frac = [float(layer) for layer in hyperparameters['HIDDEN_LAYERS'].split(',')]
        learning_rate = float(hyperparameters['LEARNING_RATE'])
        beta_1 = float(hyperparameters['BETA_1'])
        beta_2 = float(hyperparameters['BETA_2'])
        decay = float(hyperparameters['DECAY'])
        epochs = int(hyperparameters['EPOCHS'])
        batch_size=int(hyperparameters['BATCH_SIZE'])
        task = hyperparameters['TASK']
        print(pool_size, pool_stride)    
        modelo = model(input_shape = input_shape, output_dims = output_dims, 
                       conv_arch = conv_arch, conv_act = conv_act,
                       conv_filters = conv_filters, conv_dropout = conv_dropout, res_act = res_act,
                       res_filters = res_filters, res_chain=res_chain,
                       fc_dropout = fc_dropout, fc_act = fc_act,
                       filter_size = filter_size, filter_sc_size = filter_sc_size, pool_size = pool_size,
                       pool_stride = pool_stride, hl_frac = hl_frac, beta_1 = beta_1,
                       beta_2 = beta_2, decay = decay, lr = learning_rate, task = task)
        
        modelo.summary()
        data, dataframe, modelo = training(modelo, X = xtraval, Y = ytraval, epochs=epochs, 
                                                   batch_size=batch_size, 
                                                   xval = xtest,
                                                   yval = ytest, 
                                                   saveas=model_name,
                                                   verbose=1)

        ptraval = modelo.predict(xtraval_or)
        ptest = modelo.predict(xtest)
        

        modelo.save(model_name + '.h5')
        np.save(model_name + '_predtraval', ptraval)
        np.save(model_name + '_predtest', ptest)
        os.system('mv *' + model_name + '* ' + directorio)

        msetraval = mse(ytraval_or, ptraval)
        msetest = mse(ytest, ptest)
        
        maetraval = mae(ytraval_or, ptraval)
        maetest = mae(ytest, ptest)


        with open('mse_mae.txt','a') as f:
            f.write(model_name)
            f.write(',')
            f.write("%.5f" % msetraval)
            f.write(',')
            f.write("%.5f" % msetest)
            f.write(',')
            f.write("%.5f" % maetraval)
            f.write(',')
            f.write("%.5f" % maetest)
            f.close()
    
    os.system('mv mse_mae.txt ' + directorio)

    return

import sys
sys.setrecursionlimit(4000)

if sys.argv[1]:
    
    f=list(filter(None,open(sys.argv[1],'r').read().split('\n')))
    f = [i for i in f if i[0] != '#']
        
    diccio={}
    
    diccio={}
    
    for row in f:
        diccio[row.split(':')[0]] = row.split(':')[1].strip()
        
    main_function(control_file = diccio['CONTROL_FILE'],
                  output_dims = int(diccio['OUTPUT_DIMS']), 
                  test_val_frac = round(float(diccio['TEST_VAL_FRAC']),2))

