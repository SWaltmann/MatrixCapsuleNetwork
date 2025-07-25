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

    def __init__(self, directory=None):
        """Class to handle everything around the model like loading data, compiling, training, saving/loading model. 
        the directory should contain a config.json file to specify the settings,
        """        
        self.directory = directory

        self.config_path = os.path.join(directory, 'config.json')
        self.load_config(self.config_path)

        self.config['model_path'] = os.path.join(directory, 'latest_model.keras')
        self.callbacks = []

    def save_best_model(self):
        save_path = os.path.join(self.directory, 'BEST_MODEL.keras')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_categorical_accuracy',  # Save model that gets highest accuracy on validation set
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
                self.model = mcn.em_capsnet_graph(input_shape, 
                                                  _iterations=config['iterations'],
                                                  alpha=config['alpha'])
            else:  # Only other option is small
                self.model = mcn.small_em_capsnet_graph(input_shape, 
                                                        _iterations=config['iterations'],
                                                        alpha=config['alpha'])

        optimizer = self.get_optimizer()
        loss_fn = self.get_loss_fn(optimizer)
        self.save_best_model()
        self.model.compile(loss=loss_fn, 
                        optimizer=optimizer,
                        run_eagerly=config['run_eagerly'],
                        metrics=['categorical_accuracy'])
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
            ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer)
            ckpt.restore(tf.train.latest_checkpoint(self.directory)).expect_partial()
            self.set_loss_optimizer()
        else:
            raise(ValueError("No model save file was provided"))
        
    def save_model(self):
        config = self.config
        self.model.save(config['model_path'], include_optimizer=True)
        # Without this next part the optimizer states are not saved, making 
        # training start at step 0 when resumed (ie original learning rate
        # and margin)
        ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer)
        ckpt.save(os.path.join(self.directory, "latest_optimizer_ckpt"))


    def set_loss_optimizer(self):
        """SpreadLoss requires access to the optimizer's step counter (iterations)
        during training. However, this reference cannot be serialized with the model.
        This method injects the model's optimizer into the loss after loading.
        """
        self.model.loss.optimizer = self.model.optimizer
        
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
    
    def save_history(self, history):
        history_path = os.path.join(self.directory, "training_history.json")
        
        # Start with new history
        new_history = history.history

        if os.path.exists(history_path):
            # Load existing history
            with open(history_path, "r") as f:
                existing_history = json.load(f)
            
            # Append new values to existing history
            for key, new_values in new_history.items():
                if key in existing_history:
                    existing_history[key].extend(new_values)
                else:
                    existing_history[key] = new_values
            combined_history = existing_history
        else:
            # No existing history; use new one
            combined_history = new_history

        # Save the combined history
        with open(history_path, "w") as f:
            json.dump(combined_history, f)

    def get_total_epochs(self):
        """Get the number of epochs the model has already been trained for 
        by counting the entries in training_history
        """
        history_path = os.path.join(self.directory, "training_history.json")

        if os.path.exists(history_path):
            # Load existing history
            with open(history_path, "r") as f:
                history = json.load(f)
            return len(history['loss'])  # All entries are the same length, chose the loss for no reason
        else:
            # No existing history; return 0
            return 0
        
    def train(self):
        config = self.config

        self.load_dataset()
        
        self.get_model()

        print(f"\nStarting training from step {self.model.optimizer.iterations.numpy()}\n")
        # TODO: append history
        start_epoch = int(self.get_total_epochs())
        history = self.model.fit(self.train_ds,
                                 validation_data=self.val_ds,
                                 initial_epoch=start_epoch,
                                 epochs=start_epoch + config['epochs'],
                                 callbacks=self.callbacks)
        self.save_history(history)

        self.save_model()

    def test_model(self):
        self.load_dataset()
        self.get_model()
        result = self.model.evaluate(self.test_ds)
        print(result)

    

