import tensorflow as tf


class SpreadLoss(tf.keras.losses.Loss):
    def __init__(self, margin_schedule, max_steps, optimizer):
        super().__init__()
        self.schedule = margin_schedule
        self.max_steps = max_steps
        # We steal the step_counter from the optimizer
        self.optimizer = optimizer

    def call(self, y_true, y_pred):
        step = self.optimizer.iterations
        y_pred = tf.debugging.check_numerics(y_pred, message="y_pred")
       
        # print(f"current step is {current_step}")
        # print(f"y_true : {y_true}")
        # print(f"y_pred: {y_pred}")
        # # Convert {0,1} labels to {-1,1}
        # y_true = 2.0 * y_true - 1.0

        # at = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)

        # # We can use normal if/else because they have to evaluated only
        # # while building the graph
        if self.schedule == 'linear':
            margin = 0.2 + (step/self.max_steps) * 0.7
        elif self.schedule == 'sigmoid':
            margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, step / 5000.0 - 4))

        # margin = 1

        # # Target class should not be used in the sum
        # # Mask out the target class by multiplying with (1 - y_true)
        # # y_pred_masked = y_pred * (1.0 - y_true)

        # # Calculate margin loss for all classes (including target class temporarily)
        # # Broadcast at to all classes
        # # loss = tf.square(tf.maximum(0.0, margin - (at - y_pred_masked)))
        
        # loss = tf.square(tf.maximum(margin - y_true * y_pred, 0))

        # # Final loss: sum over wrong classes, then average across batch
        # return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Convert {0,1} labels to {-1,1}
        y_true = 2.0 * y_true - 1.0

        # Squared hinge loss: max(0, 1 - y_true * y_pred)^2
        per_activation_loss = tf.square(tf.maximum(0.0, margin - y_true * y_pred))

        # Sum over activations, mean over batch
        return tf.reduce_mean(tf.reduce_sum(per_activation_loss, axis=-1))
    

class CategoricalSquaredHinge(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Convert {0,1} labels to {-1,1}
        y_true = 2.0 * y_true - 1.0

        # Squared hinge loss: max(0, 1 - y_true * y_pred)^2
        per_activation_loss = tf.square(tf.maximum(0.0, 1.0 - y_true * y_pred))

        # Sum over activations, mean over batch
        return tf.reduce_mean(tf.reduce_sum(per_activation_loss, axis=-1))

