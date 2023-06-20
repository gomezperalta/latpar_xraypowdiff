#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:30:00 2021
Modified on Thu Feb 23 12:21:00 2023
@author: iG
"""
import json
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.utils as kutils
import tensorflow.keras.models as Models
import tensorflow.keras.layers as Layers
import tensorflow.keras.optimizers as Optimizers
import tensorflow.keras.initializers as Initializers
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras import backend as K

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.utils import shuffle

import copy
import time
import os

np.random.seed(3564)
initializer_seed = 10

gpu_number = 1 #### GPU number
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

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
    #x = Layers.LayerNormalization(name = 'LN_' + stage)(x)
    
    if activation:
        if activation == 'leaky_relu': x = Layers.LeakyReLU(0.15, name = activation + '_' + stage)(x)
        else: x = Layers.Activation(activation, name = activation + '_' + stage)(x)
    
    if dropout:
        x = Layers.Dropout(dropout)(x)
     
    return x, int(stage) + 1

def ResBlock(x, filters = 32, fsize_main = 4, fsize_sc = 1,
             padding = 'same', kernel_initializer='0.005',
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
                                       dropout=(fsize_sc/fsize_main)*dropout, stage=layers)
    layers -= 1
    x = Layers.Add(name='addition_' + str(layers).zfill(2))([x, x_shortcut])
    
    if activation:
        if activation == 'leaky_relu': x = Layers.LeakyReLU(0.15)(x)
        else: x = Layers.Activation(activation)(x)
     
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
        
        genhyp = [i for i in text if '\t' not in i]
        branches = [i[1:] for i in text if '\t' in i and '\t\t' not in i]
        tail = [i[2:] for i in text if '\t\t' in i]
        
        cnn_diccio = dict()
        
        bidx=[branches.index(sg) for sg in branches if 'BRANCH' in sg]+[len(branches)]
        branch_diccio = dict()

        for line in range(len(bidx)-1):
            text = branches[bidx[line]:bidx[line+1]]
            key = [entry.split(':')[0] for entry in text]
            value = [entry.split(':')[1].strip() for entry in text]
            branch_diccio[line] = {k:v for k,v in zip(key,value)}
        cnn_diccio['BRANCHES'] = branch_diccio
        
        bidx=[tail.index(sg) for sg in tail if 'CONCAT' in sg]+[len(tail)]
        tail_diccio = dict()

        for line in range(len(bidx)-1):
            text = tail[bidx[line]:bidx[line+1]]
            key = [entry.split(':')[0] for entry in text]
            value = [entry.split(':')[1].strip() for entry in text]
            tail_diccio[line] = {k:v for k,v in zip(key,value)}
        cnn_diccio['CONCAT'] = tail_diccio
        
        for line in genhyp:
            key, value = line.split(':')
            key = key.strip()
            value = value.strip()
            cnn_diccio[key] = value
    
        diccio[item] = cnn_diccio
        
    return diccio

def model_branch(input_shape = (1,1), conv_arch = list(), 
           conv_act = list(), conv_filters = list(), conv_dropout = [0.0], 
           res_act = list(), res_filters = list(), res_chain =2, 
           filter_size = [8], filter_sc_size = [1], res_dropout = list(),
           at_heads = list(), selfat_dropout = list(), 
           pool_size = 4, pool_stride = 1, serie=0):

    convs = conv_arch.count('Conv')
    resblocks = conv_arch.count('ResBlock')
    selfats = conv_arch.count('SelfAttention')
    pools = conv_arch.count('Pool')

    if len(conv_act) == 1: conv_act = [conv_act[0],]*convs
    if len(conv_filters) == 1: conv_filter = [conv_filters[0],]*convs
    if len(conv_dropout) == 1: conv_dropout = [conv_dropout[0],]*convs

    if len(res_act) == 1: res_act = [res_act[0],]*resblocks
    if len(res_filters) == 1: res_filters = [res_filters[0],]*resvlocks
    if len(filter_size) == 1: filter_size = [filter_size[0],]*resblocks
    if len(filter_sc_size) == 1: filter_sc_size = [filter_sc_size[0],]*resblocks
    if len(res_dropout) == 1: res_dropout = [res_dropout[0],]*resblocks

    if len(at_heads) == 1: at_heads = [at_heads[0],]*selfats
    if len(selfat_dropout) == 1: selfat_dropout = [selfat_dropout[0],]*selfats

    if len(pool_size) == 1: pool_size = [pool_size[0],]*pools
    if len(pool_stride) == 1: pool_stride = [pool_stride[0],]*pools

    input_layer = tensorflow.keras.Input(shape = input_shape, name = 'input_layer_' + str(serie).zfill(3))
    
    conv_count = 0
    conv_n = 0

    res_count = 0
    fn = 0
    fn_sc = 0

    selfat_count = 0

    pool_count = 0

    for item, layer in enumerate(conv_arch):
        
        if item == 0:
            
            if layer == 'Conv':
                x, layers = ConvLayer(input_layer, conv_filters[conv_count],
                              filter_size = 1, 
                              activation = conv_act[conv_count],
                              dropout = conv_dropout[conv_count],
                              stage = item + 1 + serie)
                
                conv_count += 1
                
            elif layer == 'ResBlock':
                x, layers = ResBlock(input_layer, res_filters[res_count] , 
                         fsize_main = filter_size[fn],
                         fsize_sc = filter_sc_size[fn_sc], 
                         activation = res_act[res_count],
                         dropout = res_dropout[res_count],
                         stage = item + 1 + serie, chain=res_chain)
                res_count += 1
                fn += 1
                fn_sc += 1
                conv_n += 1

        else:
            
            if layer == 'Conv':
                x, layers = ConvLayer(x, conv_filters[conv_count],
                              filter_size = 1, 
                              activation = conv_act[conv_count],
                              dropout = conv_dropout[conv_count],
                              stage = layers)
                conv_count += 1
            
            elif layer == 'Pool':
                
                x = Layers.AveragePooling1D(pool_size=pool_size[pool_count], 
                                     strides=pool_stride[pool_count])(x)
                pool_count += 1

            elif layer == 'ResBlock':
                x, layers = ResBlock(x, res_filters[res_count] , 
                                     fsize_main = filter_size[res_count],
                                     fsize_sc = filter_sc_size[res_count], 
                                     activation = res_act[res_count],
                                     dropout = res_dropout[res_count],
                                     stage = layers, chain=res_chain)
                res_count += 1
                
            elif layer == 'Permute': x = Layers.Permute((2,1))(x)
            
            elif layer == 'GAP':
                x = Layers.GlobalAveragePooling1D(name='GAP' + str(serie).zfill(3))(x)
                x = Layers.LayerNormalization(name='LNGAP' + str(serie).zfill(3))(x)
            elif layer == 'Flatten':
                x = Layers.Flatten(name = 'Flat_' + str(serie).zfill(3))(x)
                #x = Layers.LayerNormalization(name='LNFlat' + str(serie).zfill(3))(x)
    
    return Models.Model(inputs=input_layer, outputs=x)
    

def concat_models(models = list(), output_dims = 1, 
           fc_dropout = 0.0, fc_act = 'relu', hl_frac = list(), task = 'regression'):

    x = Layers.Concatenate()([i.output for i in models])

    kinit = Initializers.VarianceScaling(float(0.005), mode = "fan_avg", distribution="uniform", seed = initializer_seed)

    if output_dims != 0:

        if hl_frac[0] != 0:    
            hlnum = 2
            hidden_layers = [int(K.int_shape(x)[-1]*hl) for hl in hl_frac]
        
            for n,hl in enumerate(hidden_layers):
                if n== 0: xm = Layers.Dense(hl, name = 'FC' + str(hlnum), kernel_initializer=kinit)(x)
                else: xm = Layers.Dense(hl, name = 'FC' + str(hlnum), kernel_initializer=kinit)(xm)

                xm = Layers.LayerNormalization()(xm)
        
                if n != len(hidden_layers)-1:
                    if fc_act == 'tanh': 
                        xm = Layers.Lambda(lambda x: (2/3)*x)(xm)
                        xm = Layers.Activation(fc_act, name = fc_act + '_' + str(hlnum))(xm)
                        xm = Layers.Lambda(lambda x: (7/4)*x)(xm)
                    elif fc_act == 'leaky_relu': xm = Layers.LeakyReLU(0.15, name = fc_act + '_' + str(hlnum))(xm)
                    else:
                        xm = Layers.Activation(fc_act, name = fc_act + '_' + str(hlnum))(xm)
                hlnum += 1
            
                if fc_dropout: xm = Layers.Dropout(fc_dropout)(xm)

        xs = Layers.Dense(hl, name='FC_shortcut', kernel_initializer=kinit)(x)
        xs = Layers.LayerNormalization()(xs)
        if fc_dropout: xs = Layers.Dropout(fc_dropout)(xs)

        x = Layers.Add()([xm,xs])

        if fc_act == 'tanh':
            x = Layers.Lambda(lambda x: (2/3)*x)(x)
            x = Layers.Activation(fc_act, name = fc_act + '_res')(x)
            x = Layers.Lambda(lambda x: (7/4)*x)(x)
        elif fc_act == 'leaky_relu': x = Layers.LeakyReLU(0.15, name = fc_act + '_res')(x)
        else:
            x = Layers.Activation(fc_act, name = fc_act + '_res')(x)

        if task == 'classification' and output_dims == 1:
            x = Layers.Dense(output_dims, name='output_layer', kernel_initializer=kinit)(x)    
            output_layer = Layers.Activation('sigmoid', name='sigmoid_ouput')(x)
        
        elif task == 'classification' and output_dims != 1:
            x = Layers.Dense(output_dims, name='output_layer', kernel_initializer=kinit)(x)    
            output_layer = Layers.Activation('softmax', name='softmax_ouput')(x)
        
        else:
            output_layer = Layers.Dense(output_dims, name='output_layer', kernel_initializer=kinit)(x)    
    
    return Models.Model(inputs=[modelo.input for modelo in models], outputs=output_layer)

def model_compile(modelo, task='regression', output_dims =1, lr=1e-3, decay=1e-8, beta_1=0.9, beta_2=0.999):

    if task == 'classification' and output_dims == 1:
        modelo.compile(loss='binary_crossentropy', 
                      optimizer=Optimizers.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                      metrics=['accuracy'])
    elif task == 'classification' and output_dims != 1:
        modelo.compile(loss='categorical_crossentropy', 
                      optimizer=Optimizers.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                      metrics=['accuracy'])
    else:
        modelo.compile(loss='logcosh', #metrics =['MeanSquaredError', 'MeanAbsolutePercentageError'],
                      optimizer=Optimizers.Adamax(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,))
   
    return modelo
    
def training(model, X, Y, epochs=300, xval=np.zeros((1,1,1)), yval=np.zeros((1,1,1)), batch_size=16, saveas='modelo_nn', validation_freq=2,
             verbose=1, task='regression'):
    
    """
    Funcion copiada de patolli
    """
    modelCheckpoint=callbacks.ModelCheckpoint(str(saveas)+'.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    history = callbacks.History()
    data = model.fit(X,Y, validation_data=(xval,yval), epochs=epochs,batch_size=batch_size, callbacks=[modelCheckpoint,history],shuffle=True, verbose=verbose)

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
              
        np.save('ytraval', ytraval)
        np.save('ytest', ytest)
        
        dftraval.to_csv('dftraval.csv', index=None)
        dftest.to_csv('dftest.csv', index=None)
        
        return xtraval, xtest, dftraval, dftest, ytraval, ytest
    
    else:
        return x, None, df, None, y, None
        

def main_function(control_file = 'txt-file_name', output_dims = 1, test_val_frac=0.15):
    

    df = pd.read_csv('omint/dfset_macro.csv') #This is a support file to take interplanar distances
    minpeaks = 6   #Loading file with number of peaks higher than x-intensity
    peaks = pd.read_csv('omint/db_peaks.csv') #File containing the number of peaks with an intensity >= 5% with 50 nm
    cond0 = peaks[peaks['numpeaks'] >= minpeaks].index  #Indices that satisfy condition of minpeaks.    
    df = df.take(cond0).reset_index(drop=True)
 
    y = np.load('omint/yts2_sxv.npy') #This is first called to get the indices of the samples to work with.
    y = y[cond0]

    cond = np.argwhere((y[:,0] <= 30)*(y[:,-1] >= 1.5406))[:,0] #Compounds with interplanar distances within 60 and 1.5406 angstroms
    y = y[cond]

    y = np.load('omint/latpars.npy')
    y = y[cond0]
    y = y[cond]

    df = df.take(cond).reset_index(drop=True)

    df_temp = pd.read_csv('omint/dfset_0050.csv')
    df_temp = df_temp.take(cond0).reset_index(drop=True)
    df_temp = df_temp.take(cond).reset_index(drop=True)
    df = pd.concat((df,df_temp), ignore_index=True)

    df_temp = pd.read_csv('omint/dfset_0100.csv')
    df_temp = df_temp.take(cond0).reset_index(drop=True)
    df_temp = df_temp.take(cond).reset_index(drop=True)
    df = pd.concat((df,df_temp), ignore_index=True)

    df_temp = pd.read_csv('omint/dfset_0250.csv')
    df_temp = df_temp.take(cond0).reset_index(drop=True)
    df_temp = df_temp.take(cond).reset_index(drop=True) 
    df = pd.concat((df,df_temp), ignore_index=True)
    
    #Joining input_data
    x = np.load('omint/xset_macro.npy')
    x = x[cond0,:,:]
    x = x[cond,:,:]

    x_temp = np.load('omint/xset_0050.npy')
    x_temp = x_temp[cond0,:,:]
    x_temp = x_temp[cond,:,:]
    x = np.concatenate((x,x_temp),axis=0)

    x_temp = np.load('omint/xset_0100.npy')
    x_temp = x_temp[cond0,:,:]
    x_temp = x_temp[cond,:,:] 
    x = np.concatenate((x,x_temp),axis=0)

    x_temp = np.load('omint/xset_0250.npy')
    x_temp = x_temp[cond0,:,:] 
    x_temp = x_temp[cond,:,:] 
    x = np.concatenate((x,x_temp),axis=0)
    x_temp = ''

    #Joining output data
    y = np.concatenate((y,y,y,y))
    x2 = np.load('omint/binformula.npy')
    x2 = x2[cond0,:,:]
    x2 = x2[cond,:,:]
    x2 = np.concatenate((x2,x2,x2,x2))
    
    input_shape = [x.shape[1:], x2.shape[1:]]
    print(input_shape)
    output_dims = output_dims

    idxtest = np.random.choice(df_temp.index, size = int(test_val_frac*df_temp.shape[0]), replace=False)
    idxtest = np.concatenate((idxtest, idxtest + df_temp.shape[0], 
    							idxtest + 2*df_temp.shape[0], idxtest + 3*df_temp.shape[0]))

    idxtraval = [i for i in range(df.shape[0]) if i not in idxtest]

    xtraval = x[idxtraval]
    xtest = x[idxtest]

    xtraval2 = x2[idxtraval]
    xtest2 = x2[idxtest]
    
    ytraval = y[idxtraval]
    ytest = y[idxtest]
    
    dftraval = df.take(idxtraval).reset_index(drop=True)
    dftest = df.take(idxtest).reset_index(drop=True)
    
    #np.save('xtraval', xtraval)
    np.save('xtest', xtest)

    #np.save('xtraval2', xtraval2)
    np.save('xtest2', xtest2)

    np.save('ytraval', ytraval)
    np.save('ytest', ytest)
    
    dftraval.to_csv('dftraval.csv', index=None)
    dftest.to_csv('dftest.csv', index=None)
    df_temp = ''

    with open(control_file) as f:
        hyperparameters = json.load(f)
    
    directorio = time.ctime().replace(' ', ':').split(':')[1:]
    directorio.pop(-2)
    directorio = '_'.join(directorio)
    print('directorio', directorio, 'random state', np.random.get_state()[1][0])    
    os.system('mkdir ' + directorio)
    os.system('cp ' + control_file + ' ' + directorio)
    os.system('mv *traval* ' + directorio)
    os.system('mv *test* ' + directorio)
    
    xtraval_or = copy.deepcopy(xtraval)
    xtraval2_or = copy.deepcopy(xtraval2)
    ytraval_or = copy.deepcopy(ytraval)
    xtraval, xtraval2, ytraval = shuffle(xtraval, xtraval2, ytraval, random_state=10)
    xtraval, xtraval2, ytraval = shuffle(xtraval, xtraval2, ytraval, random_state=10)
    xtraval, xtraval2, ytraval = shuffle(xtraval, xtraval2, ytraval, random_state=10)

    model_name = hyperparameters['NAME']    
    learning_rate = float(hyperparameters['LEARNING_RATE'])
    beta_1 = float(hyperparameters['BETA_1'])
    beta_2 = float(hyperparameters['BETA_2'])
    decay = float(hyperparameters['DECAY'])
    epochs = int(hyperparameters['EPOCHS'])
    batch_size=int(hyperparameters['BATCH_SIZE'])
    task = hyperparameters['TASK']
    
    modelos = list()
    for bidx in hyperparameters['BRANCH'].keys():

        branches = hyperparameters['BRANCH'][bidx]
        bidx = int(bidx)

        globals() ['modelo_' + branches['BRANCH_NAME']] = model_branch(input_shape = input_shape[bidx], conv_arch = branches['CONV_ARCH'], 
                   conv_act = branches['CONV_ACTIVATION'], conv_filters = branches['CONV_FILTERS'], conv_dropout = branches['DROPOUT_CONV'], 
                   res_act = branches['RES_ACTIVATION'], res_filters = branches['RES_FILTERS'], res_dropout = branches['DROPOUT_RES'], 
                   res_chain = branches['RES_CHAIN'], filter_size = branches['FILTER_SIZE'], filter_sc_size = branches['FILTER_SC_SIZE'], 
                   selfat_dropout = branches['DROPOUT_ATT'], at_heads = branches['HEADS'],
                   pool_size = branches['POOL_SIZE'], pool_stride = branches['POOL_STRIDE'] , serie=bidx*200)
        
        modelos += [globals() ['modelo_' + branches['BRANCH_NAME']]]
    
    tail = hyperparameters['CONCAT']
    
    modelo = concat_models(models = modelos, output_dims = output_dims,
                    fc_dropout = tail['DROPOUT_FC'], fc_act = tail['FC_ACTIVATION'], 
                    hl_frac = tail['HIDDEN_LAYERS'], task = task)

    modelo = model_compile(modelo, task=task, output_dims = output_dims, 
                            lr=learning_rate, decay=decay, beta_1=beta_1, beta_2=beta_2)

    modelo.summary()

    print('Training set size:', xtraval.shape[0])
    print('Test set size:', xtest.shape[0])

    data, dataframe, modelo = training(modelo, X = [xtraval, xtraval2], Y = ytraval, epochs=epochs, 
                                               batch_size=batch_size, 
                                               xval = [xtest, xtest2],
                                               yval = ytest, 
                                               saveas=model_name,
                                               verbose=1)

    ptraval = modelo.predict([xtraval_or, xtraval2_or])
    ptest = modelo.predict([xtest, xtest2])
    

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
#'''
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


