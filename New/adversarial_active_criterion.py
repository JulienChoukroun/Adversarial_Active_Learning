# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.

author : mducoffe

Step 1 : deep fool as an active learning criterion
"""

##### These comments are from Julien Choukroun
##### Now we use tensorflow.keras instead of keras because we are in Tensorflow 2

import numpy as np
import tensorflow.keras.backend as K
import scipy
from contextlib import closing
import pickle as pkl
import os
from tensorflow.keras.models import Model
import tensorflow as tf
import random
import sys

from tensorflow.python.keras import backend
##### I use the cleverhans functions for the attacks
import cleverhans
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

class Adversarial_example(object):
    
    def __init__(self, model, n_channels=3, img_nrows=32, img_ncols=32, 
                 nb_class=10):
        """
        if K.image_dim_ordering() == 'th':
            img_shape = (1, n_channels, img_nrows, img_ncols)
            adversarial_image = K.placeholder((1, n_channels, img_nrows, img_ncols))
            adversarial_target = K.placeholder((1, nb_class))
            adv_noise = K.placeholder((1, n_channels, img_nrows, img_ncols))
        else:
            img_shape = (1,img_nrows, img_ncols, n_channels)
            adversarial_image = K.placeholder((1, img_nrows, img_ncols, n_channels))
            adversarial_target = K.placeholder((1, nb_class))
            adv_noise = K.placeholder((1, img_nrows, img_ncols, n_channels))
        """
        ##### The order has change, now the channel is the last parameter instead of the first parameter
        img_shape = (1, img_nrows, img_ncols, n_channels)
        adversarial_image = K.placeholder(shape=(1, img_nrows, img_ncols, n_channels))
        adversarial_target = K.placeholder(shape=(1, nb_class))
        adv_noise = K.placeholder(shape=(1, img_nrows, img_ncols, n_channels))
        
        self.model = model
        
        #####def get_weights():
            #####layers = self.model.layers
            #####norm_weights = 1
            #####for layer in layers:
                #####if hasattr(layer, 'kernel'):
                    ##### The layer.kernel.get_value() function does not work so I change it by the K.get_value(layer.kernel) function
                    #####w = np.linalg.norm(K.get_value(layer.kernel).flatten())
                    #####norm_weights*=w
            #####return norm_weights
        
        #####self.norm_weights = get_weights()
        
        self.adversarial_image= adversarial_image
        self.adversarial_target = adversarial_target
        self.adv_noise = adv_noise
        self.img_shape = img_shape
        self.nb_class = nb_class
        
        prediction = self.model.call(self.adversarial_image)
        ##### To get learning_phase in Tensorflow 2.1 with Eager execution, I have to use the symbolic_leanring_phase method from tensorflow.python.keras
        self.predict_ = K.function([backend.symbolic_learning_phase(),self.adversarial_image], K.argmax(prediction, axis=1))

    def generate(data):
        raise NotImplementedError()
        
    def predict(self,image):
        ##### I replace 0 by False because it demand a boolean
        return self.predict_([False, image])
    
    def prediction(self, image):
        return self.output_([0, image])
        
    def generate_sample(self, true_image):
        raise NotImplementedError()



class Adversarial_DeepFool(Adversarial_example):
    
    def __init__(self,  **kwargs):
        super(Adversarial_DeepFool, self).__init__(**kwargs)
        
        # HERE check for the softmax
        
        # the network is evaluated without the softmax
        # you need to retrieve the last layer (Activation('softmax'))
        #####last_dense = self.model.layers[-2].output
        #####second_model = Model(self.model.input, last_dense)
        #####loss_classif = K.mean(second_model.call(self.adversarial_image)[0, K.argmax(self.adversarial_target)])
        #####loss_classif = K.mean(second_model.call(self.adversarial_image))
        #####grad_adversarial = K.gradients(loss_classif, self.adversarial_image)
        #####with tf.GradientTape() as gtape:
            #####gtape.watch(loss_classif)
            #####gtape.watch(self.adversarial_image)
            #####second_model = Model(self.model.input, last_dense)
            #####loss_classif = K.mean(second_model.call(self.adversarial_image)
            #####grad_adversarial = gtape.gradient(target=loss_classif, sources=self.adversarial_image)
        #####self.f_loss = K.function([backend.symbolic_learning_phase(), self.adversarial_image, self.adversarial_target], loss_classif)
        #####self.f_grad = K.function([backend.symbolic_learning_phase(), self.adversarial_image, self.adversarial_target], grad_adversarial)
        
        #####def eval_loss(x,y):
            #####y_vec = np.zeros((1, self.nb_class))
            #####y_vec[:,y] +=1
            #####return self.f_loss([0., x, y_vec])
        
        #####def eval_grad(x,y):
            #####y_vec = np.zeros((1, self.nb_class))
            #####y_vec[:,y] +=1
            #####return self.f_grad([0., x, y_vec])
        
        #####self.eval_loss = eval_loss
        #####self.eval_grad = eval_grad
        

    def generate(self, data, option='fgm'):
        #perturbations=[self.generate_sample(data[i:i+1]) for i in range(len(data))]

        ##### I choose randomly an attack method between these 4 methods
        method = ['fgm', 'bim', 'pgd', 'mim']
        option = random.choice(method)
        print("OPTION:", option)
        perturbations = []
        adv_attacks = []
        i=0
        ##### I put these 3 lines in comments because now I take directly the batch of data instead of 1 data after another
        for i in range(len(data)):
            print("generate : ", i)
            r_i, x_i = self.generate_sample(data[i:i+1], option=option)
            perturbations.append(r_i)
            adv_attacks.append(x_i[0])
        print("GENERATE_SAMPLE")
        """
        # compute also the second bar
        uncertainty = []
        for i in range(len(data)):
            uncertainty.append(self.lower_bound_sample(data[i:i+1]))
        """
        index_perturbation = np.argsort(perturbations, axis=None)
        tmp = np.array(adv_attacks)
        return index_perturbation, tmp[index_perturbation]
        """
        uncertainty = np.array(uncertainty)/self.norm_weights
        index_perturbation = np.argsort(perturbations)
        
        perturbations = perturbations[index_perturbation]
        uncertainty = uncertainty[index_perturbation]
        
        N = len(data)
        sorted_index = np.arange(N)
        
        sorted_index = self.priv_sort_interval(perturbations, uncertainty, sorted_index)
        import pdb; pdb.set_trace()
        index_perturbation = index_perturbation[sorted_index]
        #return np.argsort(perturbations)
        return index_perturbation
        """
    
    def priv_sort_interval(self,array_a, array_b, sorted_index):
        
        # array_a : upper bound
        # array_b : lower bound
        N = len(array_a)
        for i in range(N-1):
            index_0 = sorted_index[i]
            index_1 = sorted_index[i+1]
            if array_a[index_0]<=array_b[index_1]:
                continue
            # array_a[i] > array_b[i+1]
            if array_b[index_1]>=array_b[index_0]:
                continue
            
            if array_b[index_0]> array_b[index_1]:
                proba = (array_b[index_0] - array_b[index_1])/(array_a[index_0] - array_b[index_1])
                if proba >=0.5:
                    
                    sorted_index[i]=index_1
                    sorted_index[i+1]=index_0
                    return self.priv_sort_interval(array_a, array_b, sorted_index)
                else:
                    continue
        return sorted_index
            
    
    def lower_bound_sample(self, true_image):
        true_label = self.predict(true_image)
        f_x = self.model.predict(true_image).flatten()
        
        score = np.inf
        index=-1
        for i in range(self.nb_class):
            if i==true_label:
                continue
            vector = np.zeros((self.nb_class))
            vector[true_label]=np.sqrt(2)
            vector[i]=-np.sqrt(2)
            score_i = np.dot(vector, f_x)
            
            if score_i <score:
                score=score_i
                index=i

        return score
         
    def generate_sample(self, true_image, option='fgm'):
        ##### I choose the attack method
        assert option in ['fgm', 'bim', 'pgd', 'mim'], ('unknown option %s',option)
        if option == 'fgm':
            return self.generate_sample_fgm(true_image)
        elif option == 'bim':
            return self.generate_sample_bim(true_image)
        elif option == 'pgd':
            return self.generate_sample_pgd(true_image)
        else:
            return self.generate_sample_mim(true_image)
            
    '''
    def generate_sample_infinity(self, true_image):

        true_label = self.predict(true_image)

        x_i = np.copy(true_image); i=0
        while self.predict(x_i) == true_label and i<10:
            other_labels = range(self.nb_class)
            other_labels.remove(true_label)
            w_labels=[]; f_labels=[]
            for k in other_labels:
                w_k = (self.eval_grad(x_i,k).flatten() - self.eval_grad(x_i, true_label).flatten())
                f_k = np.abs(self.eval_loss(x_i, k).flatten() - self.eval_loss(x_i, true_label).flatten())
                w_labels.append(w_k); f_labels.append(f_k)
            result = [f_k/(sum(np.abs(w_k))) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            r_i = (f_labels[label_adv]/(np.sum(np.abs(w_labels[label_adv]))) )*np.sign(w_labels[label_adv])
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return 0, true_image
            x_i += r_i.reshape(true_image.shape)
            #x_i = np.clip(x_i, self.mean - self.std, self.mean+self.std)
            i+=1
            
            
        adv_image = x_i
        adv_label = self.predict(adv_image)
        if adv_label == true_label:
            return np.inf, x_i
        else:
            perturbation = (x_i - true_image).flatten()
            #return np.linalg.norm(perturbation)
            return np.max(np.abs(perturbation)), x_i
    '''  
    
    '''
    def generate_sample_euclidian(self, true_image):
        true_label = self.predict(true_image)

        x_i = np.copy(true_image); i=0
        while self.predict(x_i) == true_label and i<10:
            other_labels = range(self.nb_class)
            other_labels.remove(true_label)
            w_labels=[]; f_labels=[]
            for k in other_labels:
                w_k = (self.eval_grad(x_i,k).flatten() - self.eval_grad(x_i, true_label).flatten())
                f_k = np.abs(self.eval_loss(x_i, k).flatten() - self.eval_loss(x_i, true_label).flatten())
                w_labels.append(w_k); f_labels.append(f_k)
            result = [f_k/(np.linalg.norm(w_k)) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            r_i = (f_labels[label_adv]/(np.linalg.norm(w_labels[label_adv])**2) )*w_labels[label_adv]
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return np.inf, true_image
            x_i += r_i.reshape(true_image.shape)
            #x_i = np.clip(x_i, self.mean - self.std, self.mean+self.std)
            i+=1
            
            
        adv_image = x_i
        adv_label = self.predict(adv_image)
        if adv_label == true_label:
            return np.inf, x_i
        else:
            perturbation = (x_i - true_image).flatten()
            return np.linalg.norm(perturbation), x_i
    '''


    # I use the Cleverhans functions. These functions return the adversarial example.
    def generate_sample_fgm(self, true_image):
        true_label = self.predict(true_image) #y_pred
        print("TRUE LABEL:", true_label)
        file = open('labels.txt', 'a')
        file.write(str(true_label))
        file.write(",     ")
        x_i = np.copy(true_image)
        i=0
        while self.predict(x_i) == true_label and i<10:
            r_i = fast_gradient_method(model_fn=self.model, x=x_i, eps=0.5, norm=np.inf, loss_fn=None, clip_min=None, clip_max=None, y=true_label, targeted=False, sanity_checks=False)
            x_i += tf.reshape(r_i, true_image.shape)
            i+=1
            print("i:", i)

        adv_image = x_i
        adv_label = self.predict(adv_image)
        print("ADV LABEL:", adv_label)
        file.write(str(adv_label))
        file.write("\n")
        file.close()
        perturbation = x_i-true_image 
        ##### I flatten the perturbation
        perturbation = tf.reshape(perturbation,[-1])
        distance = tf.norm(perturbation, ord='euclidean')
        file = open('distance.txt', 'a')
        file.write(str(distance.numpy()))
        file.write("\n")
        file.close()
        return distance, x_i


    def generate_sample_bim(self, true_image):
        true_label = self.predict(true_image) #y_pred
        print("TRUE LABEL:", true_label)
        file = open('labels.txt', 'a')
        file.write(str(true_label))
        file.write(",     ")
        x_i = np.copy(true_image)
        i=0
        while self.predict(x_i) == true_label and i<10:
            r_i = projected_gradient_descent(model_fn=self.model, x=x_i, eps=16, eps_iter=1e-2, nb_iter=10, norm=np.inf, loss_fn=None, clip_min=None, clip_max=None, y=true_label, targeted=False, rand_init=0, rand_minmax=None, sanity_checks=False)
            x_i += tf.reshape(r_i, true_image.shape)
            i+=1
            print("i:", i)

        adv_image = x_i
        adv_label = self.predict(adv_image)
        print("ADV LABEL:", adv_label)
        file.write(str(adv_label))
        file.write("\n")
        file.close()
        perturbation = x_i-true_image
        ##### I flatten the perturbation
        perturbation = tf.reshape(perturbation,[-1])
        distance = tf.norm(perturbation, ord='euclidean')
        file = open('distance.txt', 'a')
        file.write(str(distance.numpy()))
        file.write("\n")
       	file.close()
        return distance, x_i


    def generate_sample_pgd(self, true_image):
        true_label = self.predict(true_image) #y_pred
        print("TRUE LABEL:", true_label)
        file = open('labels.txt', 'a')
        file.write(str(true_label))
        file.write(",     ")
        x_i = np.copy(true_image)
        i=0
        while self.predict(x_i) == true_label and i<10:
            r_i = projected_gradient_descent(model_fn=self.model, x=x_i, eps=0.3, eps_iter=1e-2, nb_iter=10, norm=np.inf, loss_fn=None, clip_min=None, clip_max=None, y=true_label, targeted=False, rand_init=None, rand_minmax=0.3, sanity_checks=False)
            x_i += tf.reshape(r_i, true_image.shape)
            i+=1
            print("i:", i)

        adv_image = x_i
        adv_label = self.predict(adv_image)
        print("ADV LABEL:", adv_label)
        file.write(str(adv_label))
        file.write("\n")
        file.close()
        perturbation = x_i-true_image
        ##### I flatten the perturbation
        perturbation = tf.reshape(perturbation,[-1])
        distance = tf.norm(perturbation, ord='euclidean')
        file = open('distance.txt', 'a')
        file.write(str(distance.numpy()))
        file.write("\n")
       	file.close()
        return distance, x_i


    def generate_sample_mim(self, true_image):
        true_label = self.predict(true_image) #y_pred
        print("TRUE LABEL:", true_label)
        file = open('labels.txt', 'a')
        file.write(str(true_label))
        file.write(",     ")
        x_i = np.copy(true_image)
        i=0
        while self.predict(x_i) == true_label and i<10:
            r_i = momentum_iterative_method(model_fn=self.model, x=x_i, eps=16, eps_iter=0.06, nb_iter=10, norm=np.inf, clip_min=None, clip_max=None, y=true_label, targeted=False, decay_factor=1.0, sanity_checks=True)
            x_i += tf.reshape(r_i, true_image.shape)
            i+=1
            print("i:", i)

        adv_image = x_i
        adv_label = self.predict(adv_image)
        print("ADV LABEL:", adv_label)
        file.write(str(adv_label))
        file.write("\n")
        file.close()
        perturbation = x_i-true_image
        ##### I flatten the perturbation
        perturbation = tf.reshape(perturbation,[-1])
        distance = tf.norm(perturbation, ord='euclidean')
        file = open('distance.txt', 'a')
        file.write(str(distance.numpy()))
        file.write("\n")
       	file.close()
        return distance, x_i


