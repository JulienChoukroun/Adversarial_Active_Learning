# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:03:13 2017

@author: mducoffe
"""

import sys

import numpy as np
import sklearn.metrics as metrics
import argparse
import keras

from keras import backend as K
#from snapshot import SnapshotCallbackBuilder
import csv
from contextlib import closing
import os
from build_model import build_model_func
from build_data import build_data_func, getSize

import pickle
import gc

#%%
import resource
from keras.callbacks import Callback
class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)



#%%
def active_training(labelled_data, network_name, img_size,
                    batch_size=64, epochs=100, repeat=5):
    
    x_L, y_L = labelled_data 
    
    # split into train and validation
    
    N = len(y_L)
    n_train = (int) (N*0.8)

    batch_train = min(batch_size, len(x_L))

    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        # shuffle data and split train and val
        index = np.random.permutation(N)
        x_train , y_train = (x_L[index[:n_train]], y_L[index[:n_train]])
        x_val , y_val = (x_L[index[n_train:]], y_L[index[n_train:]])
        model = build_model_func(network_name, img_size)
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        hist = model.fit(x_train, y_train, 
             batch_size=batch_train, epochs=epochs,
             callbacks=[earlyStopping],
             shuffle=True,
             validation_data=(x_val, y_val),
             verbose=0)

        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        if loss < best_loss:
            best_loss = loss;
            best_model = model

    del model
    del hist
    del loss
    del acc
    i=gc.collect()
    while(i!=0):
        i=gc.collect()
    return best_model

#%%
def evaluate(model, percentage, test_data, nb_exp, repo, filename):
    x_test, y_test = test_data
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(nb_exp), str(percentage), str(acc)])
         
    #return query, unlabelled_pool

#%%
def get_weights(model):
    layers = model.layers
    weights=[]
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            weights+=[elem.get_value() for elem in weights_layer]
    return weights
    
def load_weights(model, weights):
    layers = model.layers
    index=0
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            for elem in weights_layer:
                elem.set_value(weights[index])
                index+=1
    return model
                
                
def loading(repo, filename, num_sample, network_name, data_name):
    # check if file exists
    img_size = getSize(data_name) # TO DO
    model=build_model_func(network_name, img_size)
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    if (os.path.isfile(os.path.join(repo, f_weights)) and \
        os.path.isfile(os.path.join(repo, f_l_data)) and \
        os.path.isfile(os.path.join(repo, f_u_data)) and \
        os.path.isfile(os.path.join(repo, f_t_data))):
        
        
        
        with closing(open(os.path.join(repo, f_weights), 'rb')) as f:
            weights = pickle.load(f)
            model = load_weights(model, weights)
            
        with closing(open(os.path.join(repo, f_l_data), 'rb')) as f:
            labelled_data = pickle.load(f)   
            
        with closing(open(os.path.join(repo, f_u_data), 'rb')) as f:
            unlabelled_data = pickle.load(f) 
            
        with closing(open(os.path.join(repo, f_t_data), 'rb')) as f:
            test_data = pickle.load(f)
    else:
        # TO DO !!!
        labelled_data, unlabelled_data, test_data = build_data_func(data_name, num_sample=num_sample)
    
    return model, labelled_data, unlabelled_data, test_data
    
def saving(model, labelled_data, unlabelled_data, test_data, repo, filename):
    weights = get_weights(model)
    #data = (weights, labelled_data, unlabelled_data, test_data)
    
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    
    with closing(open(os.path.join(repo, f_weights), 'wb')) as f:
        pickle.dump(weights, f)
    with closing(open(os.path.join(repo, f_l_data), 'wb')) as f:
        pickle.dump(labelled_data, f)
    with closing(open(os.path.join(repo, f_u_data), 'wb')) as f:
        pickle.dump(unlabelled_data, f)
    with closing(open(os.path.join(repo, f_t_data), 'wb')) as f:
        pickle.dump(test_data, f)

#%%

def active_selection(model, unlabelled_data, nb_data, active_method):
    assert active_method in ['uncertainty', 'egl', 'random'], ('Unknown active criterion %s', active_method)
    if active_method=='uncertainty':
        query, unlabelled_data = uncertainty_selection(model, unlabelled_data, nb_data, threshold)
    if active_method=='random':
        query, unlabelled_data = random_selection(unlabelled_data, nb_data)
    if active_method=='egl':
        query, unlabelled_data = egl_selection(model, unlabelled_data, nb_data)
        
        
    return query, unlabelled_data
    
def random_selection(unlabelled_data, nb_data):
    index = np.random.permutation(len(unlabelled_data[0]))
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]
    
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
           
def uncertainty_selection(model, unlabelled_data, nb_data):

    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)[::-1]
    
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]

    if len(labelled_data[0])==0:
        new_data = unlabelled_data[0][index_query]
        new_labels = unlabelled_data[1][index_query]
    else:
        new_data = np.concatenate([labelled_data[0], unlabelled_data[0][index_query]], axis=0)
        new_labels = np.concatenate([labelled_data[1], unlabelled_data[1][index_query]], axis=0)
    return (new_data, new_labels), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])

def egl_selection(model, unlabelled_data, nb_data):
    
    num_classes = model.get_output_shape_at(0)[-1]
    def get_gradient(model):
        input_shape = model.get_input_shape_at(0)
        output_shape = model.get_output_shape_at(0)
        x = K.placeholder(input_shape)
        y = K.placeholder(output_shape)
        y_pred = model.call(x)
        loss = K.mean(keras.losses.categorical_crossentropy(y, y_pred))
        weights = [tensor for tensor in model.trainable_weights]
        optimizer = model.optimizer
        gradient = optimizer.get_gradients(loss, weights)
    
        return K.function([K.learning_phase(), x, y], gradient)

    f_grad = get_gradient(model)
    
    def compute_egl(image):    
        # test
        grad = []
        
        for k in range(num_classes):
            y_label = np.zeros((1, num_classes))
            y_label[0,k] = 1
            grad_k = f_grad([0, image, y_label])
            grad_k = np.concatenate([np.array(grad_w).flatten() for grad_w in grad_k])
            grad.append(grad_k)
            
        grad = np.mean(grad, axis=0)
        return np.linalg.norm(grad)

    n = min(300, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    scores = [compute_egl(subset[i:i+1]) for i in range(len(subset))]
    index = np.argsort(scores)[::-1]
    index_query = subset_index[index[:nb_data]]
    index_unlabelled = np.concatenate( (subset_index[index[nb_data:]], subset_index[n:]))

    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])

#%%
def active_learning(num_sample, data_name, network_name, active_name,
                    nb_exp=0, nb_query=10, repo='test', filename='test.csv'):
    
    # create a model and do a reinit function
    tmp_filename = 'tmp_{}_{}_{}.pkl'.format(data_name, network_name, active_name)
    filename = filename+'_{}_{}_{}'.format(data_name, network_name, active_name)
    img_size = getSize(data_name)
    # TO DO filename
    
    model, labelled_data, unlabelled_data, test_data = loading(repo, tmp_filename, num_sample, network_name, data_name)

    batch_size = 32
    percentage_data = len(labelled_data[0])
    N_pool = len(labelled_data[0]) + len(unlabelled_data[0])
    print('START')
    # load data
    i=0
    while( percentage_data<=N_pool):

        i+=1
        model = active_training(labelled_data, network_name, img_size, batch_size=batch_size)
        
        query, unlabelled_data = active_selection(model, unlabelled_data, nb_query, active_name) # TO DO
        print('SUCCEED')
        evaluate(model, percentage_data, test_data, nb_exp, repo, filename)
        # SAVE
        saving(model, labelled_data, unlabelled_data, test_data, repo, tmp_filename)
        #print('SUCEED')
        #print('step B')
        i=gc.collect()
        while(i!=0):
            i = gc.collect()

        # add query to the labelled set
        labelled_data_0 = np.concatenate((labelled_data[0], query[0]), axis=0)
        labelled_data_1 = np.concatenate((labelled_data[1], query[1]), axis=0)
        labelled_data = (labelled_data_0, labelled_data_1)
        #update percentage_data
        percentage_data = len(labelled_data[0])
        
#%%
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Active Learning')

    parser.add_argument('--id_experiment', type=int, default=2, help='id number of experiment')
    parser.add_argument('--repo', type=str, default='.', help='repository for log')
    parser.add_argument('--filename', type=str, default='test_0', help='csv filename')
    parser.add_argument('--num_sample', type=int, default=10, help='size of the initial training set')
    parser.add_argument('--data_name', type=str, default='MNIST', help='dataset')
    parser.add_argument('--network_name', type=str, default='LeNet5', help='network')
    parser.add_argument('--active', type=str, default='random', help='active techniques')
    args = parser.parse_args()
                                                                                                                                                                                                                             

    nb_exp = args.id_experiment
    repo=args.repo
    filename=args.filename
    if filename.split('.')[-1]=='csv':
        filename=filename.split('.csv')[0]
        
    data_name = args.data_name
    network_name = args.network_name
    active_option = args.active
    num_sample = args.num_sample
    
    active_learning(num_sample=num_sample,
                    data_name=data_name,
                    network_name=network_name,
                    active_name=active_option,
                    nb_exp=nb_exp,
                    repo=repo,
                    filename=filename)


