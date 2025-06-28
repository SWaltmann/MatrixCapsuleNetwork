# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from utils.layers import PrimaryCaps, FCCaps, Length
from utils.tools import get_callbacks, marginLoss, multiAccuracy
from utils.dataset import Dataset
from utils import pre_process_multimnist
from models import efficient_capsnet_graph_mnist, efficient_capsnet_graph_smallnorb, efficient_capsnet_graph_multimnist, original_capsnet_graph_mnist, original_em_capsnet_graph_smallnorb
import os
import json
from tqdm.notebook import tqdm
import time


class Model(object):
    """
    A class used to share common model functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    
    Methods
    -------
    load_config():
        load configuration file
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        comute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights
    """
    def __init__(self, model_name, mode='test', config_path='config.json', verbose=True):
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.config_path = config_path
        self.config = None
        self.verbose = verbose
        self.load_config()


    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)
    

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERRROR] Graph Weights not found")
            
        
    def predict(self, dataset_test):
        return self.model.predict(dataset_test)
    

    def evaluate(self, X_test, y_test):
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)
        if self.model_name == "MULTIMNIST":
            dataset_test = pre_process_multimnist.generate_tf_data_test(X_test, y_test, self.config["shift_multimnist"], n_multi=self.config['n_overlay_multimnist'])
            acc = []
            for X,y in tqdm(dataset_test,total=len(X_test)):
                y_pred,X_gen1,X_gen2 = self.model.predict(X)
                acc.append(multiAccuracy(y, y_pred))
            acc = np.mean(acc)
        else:
            y_pred, X_gen =  self.model.predict(X_test)
            acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
        test_error = 1 - acc
        print('Test acc:', acc)
        print(f"Test error [%]: {(test_error):.4%}")
        if self.model_name == "MULTIMNIST":
            print(f"N° misclassified images: {int(test_error*len(y_test)*self.config['n_overlay_multimnist'])} out of {len(y_test)*self.config['n_overlay_multimnist']}")
        else:
            print(f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")


    def save_graph_weights(self):
        self.model.save_weights(self.model_path)



class EfficientCapsNet(Model):
    """
    A class used to manage an Efficiet-CapsNet model. 'model_name' and 'mode' define the particular architecure and modality of the 
    generated network.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file

    """
    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True):
        Model.__init__(self, model_name, mode, config_path, verbose)
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet{self.model_name}_new_train.weights.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"efficient_capsnet_{self.model_name}")
        self.load_graph()
    

    def load_graph(self):
        if self.model_name == 'MNIST':
            self.model = efficient_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode, self.verbose)
        elif self.model_name == 'SMALLNORB':
            self.model = efficient_capsnet_graph_smallnorb.build_graph(self.config['SMALLNORB_INPUT_SHAPE'], self.mode, self.verbose)
        elif self.model_name == 'MULTIMNIST':
            self.model = efficient_capsnet_graph_multimnist.build_graph(self.config['MULTIMNIST_INPUT_SHAPE'], self.mode, self.verbose)
            
    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.model_name, self.config_path)
        
        dataset_train_full, _ = dataset.get_tf_data()

        val_split = 0.1  # validation split as a fraction
        # Dataset is batched,find num of batches corresponding the val_split
        validation_size = int(dataset.train_size * val_split // self.config["batch_size"])

        dataset_train = dataset_train_full.skip(validation_size)
        dataset_val = dataset_train_full.take(validation_size)

        if self.model_name == 'MULTIMNIST':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
              loss=[marginLoss, 'mse', 'mse'],
              loss_weights=[1., self.config['lmd_gen']/2,self.config['lmd_gen']/2],
              metrics={'Efficient_CapsNet': multiAccuracy})
            steps = 10*int(dataset.y_train.shape[0] / self.config['batch_size'])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
              loss=[marginLoss, 'mse'],
              loss_weights=[1., self.config['lmd_gen']],
              metrics={'Efficient_CapsNet': 'accuracy'})
            steps=None

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config[f'epochs'], steps_per_epoch=steps,
          validation_data=(dataset_val), initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history

            
        
        
class CapsNet(Model):
    """
    A class used to manage the original CapsNet architecture.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (only MNIST provided)
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    n_routing: int
        number of routing interations
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train():
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file
    """
    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True, n_routing=3):
        Model.__init__(self, model_name, mode, config_path, verbose)   
        self.n_routing = n_routing
        self.load_config()
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"original_capsnet_{self.model_name}_new_train.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"original_capsnet_{self.model_name}")
        self.load_graph()

    
    def load_graph(self):
        self.model = original_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode, self.n_routing, self.verbose)
        
    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])
        
        if dataset == None:
            dataset = Dataset(self.model_name, self.config_path)          
        dataset_train, dataset_val = dataset.get_tf_data()   


        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
              loss=[marginLoss, 'mse'],
              loss_weights=[1., self.config['lmd_gen']],
              metrics={'Original_CapsNet': 'accuracy'})

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config['epochs'],
          validation_data=(dataset_val), batch_size=self.config['batch_size'], initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history


# For now this is just copied from the other model
# will be extended as necessary
class EMCapsNet(Model):
    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True):
        print("init...")
        Model.__init__(self, model_name, mode, config_path, verbose)
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"EM_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"EM_capsnet{self.model_name}_new_train.weights.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"EM_capsnet_{self.model_name}")
        self.load_graph()
        print("...init done")

    def load_graph(self):
        print("loading graph...")
        # Only works for SmallNORB (for now at least)
        if self.config['use_small_architecture']:
            self.model = original_em_capsnet_graph_smallnorb.small_em_capsnet_graph(self.config['SMALLNORB_INPUT_SHAPE'], self.mode)
        else:
            self.model = original_em_capsnet_graph_smallnorb.em_capsnet_graph(self.config['SMALLNORB_INPUT_SHAPE'], self.mode)
        print("...graph loaded")

    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])

        if dataset == None:
            dataset = Dataset(self.model_name, self.config_path)
 

        # This only works for the small_gpu setting, which I will be using anyway
        dataset_train_full, _ = dataset.get_tf_data()

        val_split = 0.1  # validation split as a fraction
        # Dataset is batched,find num of batches corresponding the val_split
        validation_size = int(dataset.train_size * val_split // self.config["batch_size"])

        dataset_train = dataset_train_full.skip(validation_size)
        dataset_val = dataset_train_full.take(validation_size)

        # Use this callback to change the margin of the MarginLoss
        spreadloss = SpreadLoss()
        callbacks = callbacks + [SpreadLossCallback(spreadloss)]

        if self.model_name == 'MULTIMNIST':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
              loss=[marginLoss, 'mse', 'mse'],
              loss_weights=[1., self.config['lmd_gen']/2,self.config['lmd_gen']/2],
              metrics={'Efficient_CapsNet': multiAccuracy})
            steps = 10*int(dataset.y_train.shape[0] / self.config['batch_size'])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
              loss=spreadloss, 
              metrics=['accuracy'],
              run_eagerly=False)
            steps=None

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config[f'epochs'], steps_per_epoch=steps,
          validation_data=(dataset_val), initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history
    
    def custom_train(self, dataset, initial_epoch=0):
        # This only works for the small_gpu setting, which I will be using anyway
        dataset_train_full, _ = dataset.get_tf_data()

        val_split = 0.1  # validation split as a fraction
        # Dataset is batched,find num of batches corresponding the val_split
        validation_size = int(dataset.train_size * val_split // self.config["batch_size"])

        dataset_train = dataset_train_full.skip(validation_size)
        dataset_val = dataset_train_full.take(validation_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr'])
        loss = SpreadLoss()

        # Prepare the metrics.
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        @tf.function
        def train_step(x, y, margin):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = loss(y, logits, margin=margin)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            train_acc_metric.update_state(y, logits)
            return loss_value
        
        @tf.function
        def test_step(x, y):
            val_logits = self.model(x, training=False)
            val_acc_metric.update_state(y, val_logits)

        for epoch in range(self.config[f'epochs']):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # The margin for the spread loss, linearly increases from 0.2 to 0.9
            margin = min(0.9, (epoch+2)/10)
            margin_tensor = tf.constant(margin, dtype=tf.float32)


            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(dataset_train):
                loss_value = train_step(x_batch_train, y_batch_train, margin_tensor)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * self.config['batch_size']))

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in dataset_val:
                test_step(x_batch_val, y_batch_val)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))

class CustomLoss(tf.keras.Loss):
    def call(self, y_true, y_pred):
        print(f"y_pred = {y_pred}\ny_true = {y_true}")
        loss = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
        return loss

class SpreadLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.9):
        super().__init__()
        self.margin = margin

    def __call__(self, y_true, y_pred, margin=0.2):
        y_pred = tf.debugging.check_numerics(y_pred, message="y_pred")
        # at: true class activation (shape [B, 1])
        at = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)

        # Calculate margin loss for all classes (including target class temporarily)
        # Broadcast at to all classes
        loss_per_class = tf.square(tf.maximum(0.0, margin - (at - y_pred)))
        
        # Target class should not be used in the sum
        # Mask out the target class by multiplying with (1 - y_true)
        masked_loss = loss_per_class * (1 - y_true)

        # Additional loss to prevent the model from predicting all zeroes
        mean = 5 * tf.reduce_mean(y_pred) 
        extra_loss = tf.square(1-mean)

        # Final loss: sum over wrong classes, then average across batch
        return tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1))


class SpreadLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_object):
        super().__init__()
        self.loss_object = loss_object

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 8:
            margin = (epoch + 2)/10
            self.loss_object.margin = margin
