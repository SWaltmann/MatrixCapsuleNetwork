import tensorflow as tf


class SpreadLoss(tf.keras.losses.Loss):
    def __init__(self, margin_schedule, max_steps):
        super().__init__()
        self.schedule = margin_schedule
        self.max_steps = max_steps

    def call(self, y_true, y_pred):
        y_pred = tf.debugging.check_numerics(y_pred, message="y_pred")
        step = y_pred[:, -1]          # shape [B], scalar per batch element (same value repeated)
        y_pred = y_pred[:, :-1]         # shape [B, C]
        current_step = step[0]    

        at = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)

        # We can use normal if/else because they have to evaluated only
        # while building the graph
        if self.schedule == 'linear':
            margin = 0.2 + (current_step/self.max_steps) * 0.7
        elif self.schedule == 'sigmoid':
            margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, current_step / 500.0 - 4))

        # Calculate margin loss for all classes (including target class temporarily)
        # Broadcast at to all classes
        loss_per_class = tf.square(tf.maximum(0.0, margin - (at - y_pred)))
        
        # Target class should not be used in the sum
        # Mask out the target class by multiplying with (1 - y_true)
        masked_loss = loss_per_class * (1 - y_true)

        # Final loss: sum over wrong classes, then average across batch
        return tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1))


class StepAwareLoss(tf.keras.losses.Loss):
    def __init__(self, step_var, name="step_aware_loss"):
        super().__init__(name=name)
        self.step_var = step_var

    def call(self, y_true, y_pred):
        current_step = tf.cast(self.step_var.read_value(), tf.float32)
        base_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        scaled_loss = base_loss * (1.0 + current_step / 1000.0)
        tf.print("Current training step:", current_step, "Loss:", scaled_loss)
        return scaled_loss

# Usage example:

step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

