import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self, config):
        # Not calling any function here makes it easier to unit test:)
        # Calling Dataset.get_smallnorb() returns preprocessed smallnorb dataset
        self.dataset_name = config['dataset_name']

        self.config = config

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Check some config values
        if self.dataset_name != 'SMALLNORB':
            raise(NotImplementedError('Only implemented dataset_name=SMALLNORB'))

    def get_dataset_size(self, dataset):
        size = 0
        for sample in dataset:
            size += 1
        return size
        
    def load_smallnorb(self):
        config = self.config
        
        train_ds, test_ds = tfds.load(
            config['dataset_name'],
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=False  # if True returns (image, label) instead of both images
            )

        # Count the dataset
        self.full_train_size = self.get_dataset_size(train_ds)
        print(f"Loaded {self.full_train_size}/24300 samples")

        return train_ds, test_ds
    
    def get_smallnorb_validation_split(self, train_ds):
        config = self.config

        if config['validation_split'] == 'loio':
            # Filter out instance 4 
            # Train data contains instances 4, 6, 7, 8, 9
            val_ds = train_ds.filter(lambda x: tf.equal(x["instance"], 4))
            val_size = self.get_dataset_size(val_ds)
            train_ds = train_ds.filter(lambda x: tf.not_equal(x["instance"], 4))
            train_size = self.get_dataset_size(train_ds)

            # Check if the two sets at least contain all samples
            if val_size+train_size!=self.full_train_size: 
                assert(ValueError("Validation + Test set != full training data"))

            # Use .take() to let tf know the size of the dataset (doesn't work)
            val_ds = val_ds.take(val_size)
            train_ds_ = train_ds.take(train_size)

        else:
            val_size = int(config['val_fraction'] * self.full_train_size)
            train_ds_ = train_ds.skip(val_size)
            val_ds = train_ds.take(val_size)

        print(f"Validation size is {val_size}")

        return train_ds_, val_ds
    
    def preprocess_smallnorb_general(sample, test_data=False):
        im1 = sample['image']
        im2 = sample['image2']
        
        # Concatenate both images for the model
        images = tf.concat((im1, im2), -1, name="combine_images")

        # Downsample to 48x48
        size = 48
        images = tf.image.resize(images, [size, size])

        # Normalize image to have zero mean and unit variance
        images = tf.image.per_image_standardization(images)

        # Create random patches for training data, and add noise
        # Create centered patches for test data (without noise)
        patch = 32  # Size of the patch
        if not test_data:
            # need ints for slicing
            corner = tf.random.uniform(shape=(2,), minval=0, maxval=size-patch, dtype=tf.int32)
            x, y = corner[0], corner[1]
            images = images[y:y+patch, x:x+patch, :]

            # Add noise
            images = tf.image.random_brightness(images, 0.4)
            images = tf.image.random_contrast(images, 0.0, 3.0)
        else:
            low = (size-patch)//2
            high = (size+patch)//2
            images = images[low:high, low:high, :]

        y = tf.one_hot(sample['label_category'], 5)
        
        return images, y
    
    def preprocess_smallnorb(self, train_ds, val_ds, test_ds):

        def preprocess_smallnorb_train(sample):
            return self.preprocess_smallnorb_general(sample, test_data=False)
        
        def preprocess_smallnorb_test(sample):
            return self.preprocess_smallnorb_general(sample, test_data=True)
        
        batch_size = self.config['batch_size']
        self.train_ds = train_ds.map(preprocess_smallnorb_train).batch(batch_size)
        # valiidation is taken from training data, but preprocessed as test
        # to better mimck the test data
        self.val_ds = val_ds.map(preprocess_smallnorb_test).batch(batch_size)
        self.test_ds = test_ds.map(preprocess_smallnorb_test).batch(batch_size)

        return self.train_ds, self.val_ds, self.test_ds

    def get_smallnorb(self):
        # If any dataset doesn't exist yet, go get it. Else just return the already
        # preprocessed datasets.
        if (self.test_ds is None) or (self.val_ds is None) or (self.test_ds is None):
            train_ds, test_ds = self.load_smallnorb()
            val_ds, train_ds = self.get_smallnorb_validation_split(train_ds)
            return self.preprocess_smallnorb(train_ds, val_ds, test_ds)
        else:
            return self.train_ds, self.val_ds, self.test_ds

