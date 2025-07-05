import numpy as np
import tensorflow as tf
from utils.dataset import Dataset
from utils import loss_functions
from models import matrix_capsule_networks as mcn
import os
import json
from tqdm.notebook import tqdm
import time


class MatrixCapsuleNetwork:

    def __init__(self, config_path='config.json'):
        """Class to handle everything round the model like loading data, compiling, training, saving/loading model. 
        args: config is read from config.json to get the settings. If no save_path is provided we will create a new model, else we load from the save path."""        
        self.load_config(config_path)  # Create self.config
        self.callbacks = []
        self.config_path = config_path

    def save_best_model(self):
        save_path = 'BEST_'+ self.config['model_path']
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',  # Save model that gets highest accuracy on validation set
            save_best_only=True,
            mode='max')
        self.callbacks.append(checkpoint)

    def load_config(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def load_dataset(self):
        # Dataset is selected through config.json
        self.dataset = Dataset(self.config)
        train, val, test = self.dataset.get_data()
        self.train_ds = train
        self.val_ds = val
        self.test_ds = test
        
        return self.dataset

    def get_model_architecture(self):
        input_shape = self.dataset.input_shape
        config = self.config

        if config['dataset_name'] == 'smallnorb':

            if config['model_size'] == 'full':
                self.model = mcn.em_capsnet_graph(input_shape, _iterations=config['iterations'])
            else:  # Only other option is small
                self.model = mcn.small_em_capsnet_graph(input_shape, _iterations=config['iterations'])

        # Next time we call this same config_file, we will load the
        # model instead of training from scratch
        config['use_pretrained'] = True
        with open(self.config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

        return self.model
    
    def load_model(self):
        config = self.config
        save_path = config['model_path']
        if save_path is not None:
            self.model = tf.keras.models.load_model(save_path)
        else:
            raise(ValueError("No model save file was provided"))
        
    def save_model(self):
        config = self.config
        self.model.save(config['model_path'])
        
    def get_model(self):
        if self.config['use_pretrained']:
            self.load_model()
        else:
            self.get_model_architecture()

    def get_optimizer(self):
        config = self.config
        if config['lr_decay_rate'] > 0:
            lr= tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=config['learning_rate'],
                decay_steps=config['lr_decay_steps'],
                decay_rate=config['lr_decay_rate']
            )
        else:
            lr = config['learning_rate']

        if config['optimizer'] == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        elif config['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        return optimizer
    
    def get_loss_fn(self, optimizer):
        config = self.config
        if config['loss_function'] == 'cross_entropy':
            # Our logits behave like activations, so we can use the 
            # default from_logits=False
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        elif config['loss_function'] == 'spread_loss':
            sched = config['margin_schedule']
            loss_fn = loss_functions.SpreadLoss(sched, 
                                                config['max_margin_steps'],
                                                optimizer)
        elif config['loss_function'] == 'squared_hinge':
            loss_fn = loss_functions.CategoricalSquaredHinge()
        else:
            raise(NotImplementedError("This loss function is not implemented"))
        
        return loss_fn
    
    def train(self):
        config = self.config

        self.load_dataset()
        self.get_model()

        optimizer = self.get_optimizer()
        loss_fn = self.get_loss_fn(optimizer)
        self.save_best_model()
        self.model.compile(loss=loss_fn, 
                           optimizer=optimizer,
                           run_eagerly=config['run_eagerly'],
                           metrics=['categorical_accuracy'])
        history = self.model.fit(self.train_ds,
                                 validation_data=self.val_ds,
                                 epochs=config['epochs'],
                                 callbacks=self.callbacks)
        # TODO: make this a seperate function and store everything in its own little directory
        with open(f"training_history{config['run_name']}.json", "w") as f:
            json.dump(history.history, f)

        self.save_model()

    

