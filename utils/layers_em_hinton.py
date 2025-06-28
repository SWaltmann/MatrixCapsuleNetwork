import tensorflow as tf
import numpy as np
import math


class ReLUConv(tf.keras.layers.Layer):
    def __init__(self, A=32, kernel_size=5, stride=2, **kwargs):
        super(ReLUConv, self).__init__(**kwargs)

        # Settings
        self.num_channels = A  # num_channels is more descriptive than 'A'
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = None
    
    def build(self, input_shape):
        # in_channels = input_shape[-1]  # Check for grayscale/color
        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_channels, 
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='valid',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(.05),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=5e-2),
            bias_initializer=tf.keras.initializers.Constant(0.1)
        )
        self.built = True

    def call(self, inputs):
        return self.conv(inputs)


class PrimaryCaps(tf.keras.layers.Layer):
    """
    All 'magic numbers' are either from 'Matrix Capsules with EM Routing', or
    taken from the corresponding repo at google-research's github:
    https://github.com/google-research/google-research/tree/master/capsule_em
    """
    def __init__(self, B=32, out_atoms=16, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = B  # number of output capsule
        self.out_atoms = out_atoms  # number of values in pose matrix (16 or 9)
        self.conv = None
        self.kernel_size=1
        self.stride=1
        self.sqrt_atoms = np.sqrt(out_atoms)

    def build(self, input_shape):
        self.conv_dim = input_shape[-1]  # = A
        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_capsules * (self.out_atoms+1),  # 1 for the activation
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(.0000002),
            use_bias=False
        )
        # # Manually construct the kernel out of two tensors with different 
        # # initialization parameters
        # self.conv.build(input_shape)  # build layer so that we can override kernel

        # # Create the kernels
        # pose_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)
        # pose_kernel = pose_initializer(shape=[self.kernel_size, 
        #                                       self.kernel_size, 
        #                                       input_shape[-1], 
        #                                       self.out_atoms * self.num_capsules])

        # activation_initializer = tf.keras.initializers.TruncatedNormal(stddev=3.0)
        # act_kernel = activation_initializer(shape=[self.kernel_size, 
        #                                            self.kernel_size, 
        #                                            input_shape[-1], 
        #                                            1 * self.num_capsules])
        
        # kernel = tf.concat([pose_kernel, act_kernel], axis=-1)
        # self.conv.kernel.assign(kernel)

        self.built = True

    def call(self, inputs):
        # Input shape: (batch_size, H, W, channels)
        conv_output = self.conv(inputs)
        # cont_output.shape = (batch_size, H, W, channels), 
        #   with channels size: num_capsules*(1+num_atoms)

        # Split the pre-activation from the pose. So now there are two tensors 
        # with channel sizes 1 and num_atoms, respectively
        poses, pre_activations = tf.split(conv_output,
                                          [self.out_atoms*self.num_capsules,
                                           self.num_capsules],
                                           axis=-1)
        
        # Instead of a big out_atoms*num_capsule dimension, we reshape it into 
        # a tensor of shape (num_capsules, sqrt, sqrt) so each capsule has its 
        # own pose matrix:
        # Example (N, W, H, 512) -> (N, W, H, 32, 4, 4)
        poses_shape = tf.shape(poses)
        N, H, W = poses_shape[0], poses_shape[1], poses_shape[2] 
        poses_reshaped = tf.reshape(poses, [N, H, W, 
                                            self.num_capsules, 
                                            self.sqrt_atoms,
                                            self.sqrt_atoms])
        
        # Make sure the activation shape matches for broadcasting:
        #   (N, W, H, C) -> (N, W, H, C, 1, 1)
        pre_act_reshaped = tf.expand_dims(pre_activations, -1)
        pre_act_reshaped = tf.expand_dims(pre_act_reshaped, -1)

        # Compute activations by passing through sigmoid:
        activations_reshaped = tf.math.sigmoid(pre_act_reshaped)

        return poses_reshaped, activations_reshaped


class ConvCaps(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, C=32, stride=1, **kwargs):
        """Kernel is shaped (kernel_size, kernel_size)"""
        super(ConvCaps, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_capsules = C  # number of output capsules
        self.stride = stride
        self.conv = None

    def build(self, input_shape):
        # input shape = (N, H, W, capsules_in, sqrt_atom, sqrt_atom)
        #   Example: (N, H, W, 32, 4, 4) in case of 32 capsules with 4x4 matrix
        self.pose_shape_in = input_shape[0]
        self.act_shape_in = input_shape[1]

        self.in_capsules = self.pose_shape_in[3]
        self.sqrt_atom = self.pose_shape_in[-1]
        
        self.pose_kernel = self.add_weight(shape=(self.kernel_size,
                                                  self.kernel_size,
                                                  self.in_capsules,
                                                  self.out_capsules,
                                                  self.sqrt_atom,
                                                  self.sqrt_atom),
                                           initializer=tf.keras.initializers.TruncatedNormal(stddev=0.3),
                                           name = "pose_kernel",)
        self.built = True

    def call(self, inputs):
        poses, activations = inputs

        votes_out = self.conv_caps(poses, activations=False)
        act_out = self.conv_caps(activations, activations=True)

        votes_out = tf.debugging.check_numerics(votes_out, message="votes in ConvCaps is messed up")
        act_out = tf.debugging.check_numerics(act_out, message="acts_out in ConvCaps is messed up")

        return votes_out, act_out
        
    
    def conv_caps(self, _input, activations=True):
        """The poses and activations undergo essentially the same operations
        so to prevent duplicated code we just use call() as a wrapper to pass
        them individually through this method which does the actual work.

        activations arg is used to specificy wether activations are passed
        (True) or poses (False). Determines which kernel to use:)
        """
        # Reshape into a 4D tensor: (N, H , W, caps_in*in_atoms)
        # This has to be done because extract_patches() only takes 4D tensors
        _shape = tf.shape(_input)

        input_reshaped = tf.reshape(_input, (_shape[0], _shape[1], _shape[2], 
                                            _shape[3]*_shape[4]*_shape[5]))
        # Taking patches is similar to applying convolution, We cannot use 
        # Conv2D (for example) because it does not do matrix multiplication
        patches = tf.image.extract_patches(input_reshaped, 
                                                 [1, self.kernel_size, self.kernel_size, 1],
                                                 [1, self.stride, self.stride, 1],
                                                 [1, 1, 1, 1],
                                                 'VALID')

        # Output shape based on 'valid' padding !
        out_height = (_shape[1] - self.kernel_size) // self.stride + 1
        out_width = (_shape[2] - self.kernel_size) // self.stride + 1

        # Reshape the patches back into pose matrices
        patches = tf.reshape(patches,
                                  (_shape[0], out_height, out_width,  # N, H, W
                                  self.kernel_size, self.kernel_size,
                                  _shape[-3], _shape[-2], _shape[-1]))
        
        # Hinton now transposes and reshapes the patches for optimal performance
        # TF2 fixes this behind the scenes, so we keep it in a shape that makes
        # more sense to me:)

        if activations:
            # We are working on the activations - they are not multiplied
            # by the kernel
            return tf.expand_dims(patches, -3)  # For consistency with the poses 
     
        # Each patch must be mulitplied by the kernel
        # The kernel should matmul each pose matrix with a unique transformation matrix
        # Kernel should have the same shape as 1 patch, but with additional channels
        # for each output capsule (defined in build() method

        # Looks terrible, but does the matrix multiplication between kernel and patches
        # Cannot repeat indices, so I use xy for kernel (instead of more intuitive k)
        # same for pose matrix (mnp instead of p)
        #   b=batch, h=height, w=width, xy=kernel*kernel, i=in_capsules, o=out_capsules  
        #   mnp=pose_matrix (4*4)
        matrices = tf.einsum('bhwxyimn,xyiopn->bhwxyiomp', patches, self.pose_kernel)
        # shape = tf.shape(patches)  # shape is a Tensor
        # b, h, w, k, _, i, a, _ = tf.unstack(shape)
        # kernel = self.pose_kernel
        # o = tf.shape(kernel)[3]
        # # First I reshape both to b, h, w, k, k, i, o, a, a
        # # That is so that both can be broadcasted to the correct numbers
        # # For the missing dims I put a 1 now, to be broadcasted later
        # patches_reshape = tf.reshape(patches, (b, h, w, k, k, i, 1, a, a))
        # kernel_reshape  = tf.reshape(kernel,  (1, 1, 1, k, k, i, o, a, a,))

        # # Then we broadcast them to the same shape. Now we have the same amount
        # # of tensors in both the patches and the kernels
        # patch = tf.broadcast_to(patches_reshape, (b, h, w, k, k, i, o, a, a))
        # kern = tf.broadcast_to(kernel_reshape, (b, h, w, k, k, i, o, a, a))

        # # Now we reshape them, so that we are left with many matrices that can be matmulled
        # batch_patch = tf.reshape(patch, (b*h*w*k*k*i*o, a, a))
        # batch_kern = tf.reshape(kern, (b*h*w*k*k*i*o, a, a))

        # matrices = tf.matmul(batch_patch, batch_kern)
        # matrices = tf.reshape(matrices, (b, h, w, k, k, i, o, a, a))

        #TODO: I am unsure this does the correct multiplications could be summing things
        return matrices


class ClassCaps(tf.keras.layers.Layer):
    def __init__(self, position_grid, out_atoms=16, capsules_out=5, **kwargs):
        super(ClassCaps, self).__init__(**kwargs)
        # Position grid has to be tracked globally for the full model
        # so there is a seperate function to track that external to the layers
        # That function is in the model definition
        self.position_grid = position_grid
        self.out_atoms = out_atoms
        self.caps_out = capsules_out
        # Build grid from position_grid (numpy)
        xv, yv = self.position_grid  # Each: [H, W]
        grid = np.stack([xv, yv], axis=-1)  # [H, W, 2]
        self.grid = tf.convert_to_tensor(grid, dtype=tf.float32)

        

    def build(self, input_shape):  
        # input shape = (N, H, W, capsules_in, sqrt_atom, sqrt_atom)
        #   Example: (N, H, W, 32, 4, 4) in case of 32 capsules with 4x4 matrix
        self.pose_shape_in = input_shape[0]
        self.in_sqrt_atoms = self.pose_shape_in[-1]
        self.act_shape_in = input_shape[1]
        self.caps_in = self.pose_shape_in[3]

        

        self.weight = self.add_weight(shape=(self.caps_in, self.caps_out, self.in_sqrt_atoms, self.in_sqrt_atoms),
                                       initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                                       name='weights')

        self.built = True

    def call(self, inputs):
        poses, activations = inputs
        # First compute all the votes
        votes = tf.einsum('bhwimn,ionp->bhwiomp', poses, self.weight)
        # Then add the coordinates to all the votes
        votes = self.coordinate_addition(votes)
        
        # That's it I guess? now Routing
        # votes.shape = [batch, height, width, in_caps, out_caps, atom, atom]

        # To make routing work, add a 1x1 kernel to it
        # Shape should be [batch, H, W, 1, 1, in, out, atom, atom]
        activations = self.add_kernel(activations) 
        votes, activations = self.add_kernel(votes), tf.expand_dims(activations, -3)  # add out_caps 

        # We treat these votes as if they are all coming from the center. Instead of
        # a [H, W] grid, we act as if there is just 1 position with H*W votes in it.
        # In that way, routing should still work:)

        # Routing expects
        # [batch, height, width, kernel, kernel, in, out, atom, atom]
        # So we create a
        vs = tf.shape(votes)
        b, h, w, i, o, a= vs[0], vs[1], vs[2], vs[5], vs[6], vs[7]

        votes = tf.reshape(votes, (b, 1, 1, 1, 1, h*w*i, o, a, a))
        acts = tf.reshape(activations, (b, 1, 1, 1, 1, h*w*i, 1, 1, 1))
        votes = tf.debugging.check_numerics(votes, message="votes in ClassCaps is messed up")
        acts = tf.debugging.check_numerics(acts, message="acts in ClassCaps is messed up")
        return votes, acts

    def add_kernel(self, tensor):
        # Add dimensions after batch, height, width
        return tf.expand_dims(tf.expand_dims(tensor, 3), 3)


    def coordinate_addition(self, poses):
        # poses: [batch, H, W, caps_in, sqrt_atom, sqrt_atom]
        batch_size, H, W = tf.shape(poses)[0], tf.shape(poses)[1], tf.shape(poses)[2]
        sqrt_atom = tf.shape(poses)[-1]

        grid = self.grid

        # Create zero matrices: [H, W, sqrt_atom, sqrt_atom]
        zeros = tf.zeros((H, W, sqrt_atom, sqrt_atom), dtype=poses.dtype)

        # Extract x and y
        x = grid[..., 0]  # [H, W]
        y = grid[..., 1]  # [H, W]


        # Build scatter indices
        hw = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing="ij"), axis=-1)  # [H, W, 2]
        hw_flat = tf.reshape(hw, [-1, 2])  # [H*W, 2]

        # Make sure fill() gets a value and not a tensor that does not exist
        sqrt_atom_val = tf.shape(poses)[-1]
        fill_val = tf.cast(sqrt_atom_val - 1, tf.int32)

        x_idx = tf.concat([hw_flat, tf.zeros((H*W, 1), tf.int32), tf.fill((H*W, 1), fill_val)], axis=1)
        y_idx = tf.concat([hw_flat, tf.ones((H*W, 1), tf.int32), tf.fill((H*W, 1), fill_val)], axis=1)

        # Flatten values
        x_val = tf.reshape(x, [-1])
        y_val = tf.reshape(y, [-1])

    
        all_indices = tf.concat([x_idx, y_idx], axis=0)
        all_values = tf.concat([x_val, y_val], axis=0)

        # Scatter into zeros: result is [H, W, sqrt_atom, sqrt_atom]
        coord = tf.tensor_scatter_nd_update(zeros, all_indices, all_values)

        # Reshape to broadcast: [1, H, W, 1, 1, sqrt_atom, sqrt_atom]
        # To account for batch, in_caps, out_caps
        coord = tf.reshape(coord, [1, H, W, 1, 1, sqrt_atom, sqrt_atom])

        # Add to poses (broadcasted over batch and capsules)
        return poses + coord



class EMRouting(tf.keras.layers.Layer):
    """ Please note that although this is implemented as a seperate layer, it
    should be viewed as part of the previous layer. ConvCaps handles only the
    pose matrices and selects the corresponding activations without altering
    them. This layer takes the activations and poses and finds the new 
    activations.
    """
    def __init__(self, mean_data=1, iterations=2, min_var=0.0005, final_beta=0.01, epsilon_annealing=False, **kwargs):
        super(EMRouting, self).__init__(**kwargs)
        self.iterations = iterations
        self.min_var = min_var
        self.final_lambda = final_beta
        self.verbose = False
        self.epsilon = 1e-7
        # From Gritzman. Has to be manually calcualted for now
        self.mean_data = mean_data  # if 1, this does nothing

        # Slowly decrease epsilon, to make model learn faster and be more stable
        self.epsilon_annealing = epsilon_annealing
        if epsilon_annealing:
            self.call_count = self.add_weight(
                name="call_count",
                initializer="zeros",
                dtype=tf.int64,
                trainable=False
            )  
            self.decay_steps = 50000
            self.epsilon_start = 1e-7
            self.epsilon_end = 1e-14

    def compute_epsilon(self):
        step = tf.cast(self.call_count, tf.float32)
        decay_ratio = tf.minimum(1.0, step / self.decay_steps)
        epsilon = self.epsilon_start * (1 - decay_ratio) + self.epsilon_end * decay_ratio
        return epsilon

    def build(self, input_shape):
        # Pose input:
        # Shape: [batch, height, width, kernel, kernel,
        #         in_caps, out_caps, sqrt_atom, sqrt_atom] 
        # Intuition: For every batch, at every grid position (height x width), 
        #   there is a kernel (kernel x kernel) consisting of in_caps number of 
        #   capsules. Each of those capsules votes for each of the out_caps number 
        #   of output capsules, and each vote is in the form of a 
        #   pose matrix (sqrt_atom x sqrt_atom)

        # Activation input:
        # Shape: [batch, height, width, kernel, kernel
        #         in_caps, 1, 1, 1]
        # activations have a similar shape, but lack the out_caps dimension, since 
        # only the lower-level capsules have an activation at this point (we are
        # calculating the higher level activations in this layer). There also is 
        # only 1 value per lower-level capsule, we keep singleton dimensions for
        # easier broadcasting later:)

        self.pose_shape_in = input_shape[0]
        ps = self.pose_shape_in
        b, h, w, i, o, a= ps[0], ps[1], ps[2], ps[5], ps[6], ps[7]
        self.act_shape_in = input_shape[1]


        p_shape = self.pose_shape_in

        # Initialize biases (using Hinton's setting)
        # activation_bias in Hinton's implementation
        self.beta_a = self.add_weight(
            shape=(1, 1, 1, 1, 1, 1, p_shape[6], 1, 1),  # Each higher-level capsule has its own activation cost
            initializer=tf.constant_initializer(0.5),
            name='activation_bias'
        )
        # sigma_bias in Hinton's implementation
        self.beta_u = self.add_weight(
            shape=(1, 1, 1, 1, 1, 1, p_shape[6],1, 1),  # Each higher-level capsule has its own activation cost
            initializer=tf.constant_initializer(0.5),
            name='sigma_bias'
        ) 
        self.built = True

        
    def call(self, inputs):

        if self.epsilon_annealing:
            epsilon = self.compute_epsilon()
            # Update the call counter
            # This probably also updates during validation and testing...
            # Not sure how to fix that. Oh well :)
            self.call_count.assign_add(1)
        else:
            epsilon = self.epsilon
        
        votes, activations = inputs
        votes = tf.debugging.check_numerics(votes, message="VOTES ARE FUCKEDD")
        activations = tf.debugging.check_numerics(activations, message="ACTS ARE FUCKEDD")

        vs = tf.shape(votes)
        b, h, w, k, i, o, a = vs[0], vs[1], vs[2], vs[3], vs[5], vs[6], vs[7]

        # Initialize empty tensors as input for 1st iter of the routing loop
        self.out_activations = tf.zeros((b, h, w, 1, 1, 1, o, 1, 1))
        self.out_poses = tf.zeros((b, h, w, 1, 1, 1, o, a, a))
        # post in Hinton's implementation
        self.R_ij = tf.nn.softmax(tf.zeros((b, h, w, 1, 1, i, o, 1, 1)), axis=6)
        self.R_ij_prev = self.R_ij


        # Perform routing
        for i in range(self.iterations - 1):
            if self.verbose:
                print(f"STARTING ITERATION {i+1}")
            self.m_step(activations, votes, i, epsilon)
            self.e_step_log(votes, epsilon)
            if self.verbose:
                print("\n\n")
        # Last routing iteration only requires the m-step
        if self.verbose:
            print(f"FINAL ITERATION ({i+2})")
        self.m_step(activations, votes, self.iterations, epsilon)
        if self.verbose:
            print("-"*60)
            print("\n\n")

        # Remove the singleton dimsionsions that are left over from the kernel 
        # and in_caps so that the shape matches that of the original input to 
        # the conv_caps layer before this
        self.out_poses = tf.squeeze(self.out_poses, axis=[3,4,5])
        self.out_activations = tf.squeeze(self.out_activations, axis=[3,4,5])

        poses = tf.debugging.check_numerics(self.out_poses, message="votes after Routing is messed up")
        acts = tf.debugging.check_numerics(self.out_activations, message="acts after Routing is messed up")
    
        return poses, acts


    def m_step(self, a_i, V_ij, i, epsilon):
        # Hinton names the variables differently in his code than 
        # in the paper. We stick to the paper, but this is the translation:
        #   my var - Hinton's var
        #   R_ij - posterior
        #   a_i - activation
        #   V_ij - wx

        #   the 'masses' arg in Hintons code is unnecessary (it gets redefined)

        # I know that _j, _j, _h are just indices to a specific element. I kept
        # them in the variable names so that it is easier to match to the pseudo 
        # code in the paper.

        R_ij = self.R_ij

        # vote_conf in Hinton's implementation
        alpha = 0.2
        R_ij = (R_ij * a_i) * (1 - alpha) + alpha * self.R_ij_prev
        self.R_ij_prev = R_ij
        # All values are the same per capsule, just repeated over multiple kernels
        if self.verbose:
            print("R_ij * a_i is:")
            print(R_ij)
        # We need Sum_i(R_ij) multiple times so we'll store it:
        # masses in Hinton's implementation
        sum_R_ij = tf.reduce_sum(R_ij, axis=[3,4,5], keepdims=True)
        if self.verbose:
            print("sum_R_ij is:")
            print(sum_R_ij)
        # V_ij are the votes, shaped:    [batch, height, width, kernel, kernel, 
        #                                 caps_in, caps_out, sqrt_atom, sqrt_atom]
        # R_ij is shaped:                [batch, height, width, kernel, kernel,
        #                                 caps_in, caps_out, 1, 1]

        # Each of the 16 pose values must be multiplied by the same R_ij
        # Since we were smart about the shapes they broadcast nicely

        # The summation should be done over ALL lower-level capsules. For
        # each higher-level capsule there are caps_in lower-level votes
        # for each position in the kernel. So we must sum over dimensions
        # kernel, kernel AND caps_in

        # It should result in a mu for each value of the pose matrix, ie shape:
        #   [batch, heigth, width, 1, 1, 1, caps_out, sqrt_atom, sqrt_atom]
        # This is kind of a weighed average of the votes - but it is averaged
        # in a weird way (averaging each value seperately instead of treating 
        # the pose matrix as 3D thing). Will not result in a valid pose mat.

        # So now mu_jh[:, :, :, :, :, : j, h1, h2] is mu_j^h from the paper
        # BTW, the paper treats the pose matrix as a vector with length 16,
        # which is why there is only 1 index for the value in the matrix 

        # preactivate_unrolled in Hinton's implementation. Hinton then
        # combines this with the old value to get 'center'. But we skip
        # that step since it is not mentioned in the paper.
        mu_jh = (tf.reduce_sum(R_ij * V_ij, axis=[3,4,5], keepdims=True) 
                / (sum_R_ij + epsilon))  # e-7 from Hinton's implementation
                # e-7 prevents numerical instability (Gritzman, 2019)
        if self.verbose:
            print("mu_jh is:")
            print(mu_jh)
        # variance in Hinton's implementation
        sigma_jh_sq = (tf.reduce_sum(R_ij * tf.pow((V_ij - mu_jh), 2), axis=[3,4,5], keepdims=True)
                       / (sum_R_ij + epsilon))
        # Make sure the variance is never 0, or super large
        # sigma_jh_sq_clipmax = tf.minimum(sigma_jh_sq, 1e9)
        # tf.print("Clipped ratio (was too large):", tf.reduce_mean(tf.cast(sigma_jh_sq != sigma_jh_sq_clipmax, tf.float32)))
        # sigma_jh_sq_clipmin = tf.maximum(sigma_jh_sq_clipmax, 1e-9)
        # tf.print("Clipped ratio (was too small):", tf.reduce_mean(tf.cast(sigma_jh_sq_clipmin != sigma_jh_sq_clipmax, tf.float32)))
       
        # sigma_jh_sq = sigma_jh_sq_clipmin
        if self.verbose:
            print("sigma_jh_sq is:")
            print(sigma_jh_sq)
        # This happens in the paper, but not in Hinton's code
        # sigma_jh = tf.math.sqrt(sigma_jh_sq)

        # Completely lost how Hinton's code relates to their paper at this point
        # Good luck figuring that out    
        cost_h = (self.beta_u - 0.5 * tf.math.log(sigma_jh_sq + epsilon)) * sum_R_ij / self.mean_data
        # tf.print(cost_h)
        if self.verbose:
            print("cost_h is:")
            print(cost_h)
        # beta in Hinton's implementation
        inverse_temp = self.final_lambda*(1-tf.pow(0.95, i+1))

        if self.verbose:
            print("inverse temperature is:")
            print(inverse_temp)

        # activation_update in Hinton's implementation (I THINK, shit's a maze imo)
        # Maybe logit is actually closer but yout guess is as good as mine
        a_j = tf.math.sigmoid(
            inverse_temp*(self.beta_a - tf.reduce_sum(cost_h, axis=[-1,-2], keepdims=True))  # Sum over values in pose matrix
              )
        if self.verbose:
            print("a_j is:")
            print(a_j)

            print("BEFORE THE SIGMOID:")
            print(inverse_temp*(self.beta_a - tf.reduce_sum(cost_h, axis=[-1,-2], keepdims=True)))
        # a_j = tf.debugging.check_numerics(a_j, message="a_j")
        # Assign everythin to the corresponding attributes 
        # Could have done that immediately but wanted to follow paper's notation
        # a_j = tf.nn.softmax(a_j, axis=6)
        self.out_activations = a_j
        self.out_poses = mu_jh
        self.sigma_jh_sq = sigma_jh_sq
        # R_ij is updated and assigned in e_step

    def e_step(self, V_ij):
        mu_jh = self.out_poses
        a_j = self.out_activations
        # This is very different from what happens in Hinton's code. Highly
        # doubt it is equivalent but theirs is so hard to follow
        exponent = -0.5*tf.reduce_sum(
                tf.pow((V_ij - mu_jh), 2) / self.sigma_jh_sq, axis=[-1,-2], keepdims=True
            )
        
        # Make sure that none of the values are too large before 
        # exponentiating. This is a trick to prevent numerical instability. It 
        # is not mentioned in the paper, but it is in Hinton's code
        
        # Basically we subtract the maximum value from all values in the
        # exponent tensor per output capsule. Not 100% sure if this works
        exponent = exponent - tf.reduce_max(exponent, axis=[3,4,6], keepdims=True)

        p_j = (1
             /tf.math.sqrt(tf.reduce_prod(2*math.pi*self.sigma_jh_sq, axis=[-1,-2], keepdims=True))
            )*tf.math.exp(exponent)
        

        self.R_ij = a_j * p_j / tf.reduce_sum(a_j * p_j, axis=[3,4,6], keepdims=True)
        

    def e_step_log(self, V_ij, epsilon):
        """Compute the e-step in log space to prevent NaN from small (<e-2) inputs"""
        mu_jh = self.out_poses
        a_j = self.out_activations

        # Compute the log probability (log p_j)
        log_p_j = -tf.reduce_sum(
            tf.math.log(2 * math.pi * self.sigma_jh_sq + epsilon) +
            tf.pow((V_ij - mu_jh), 2) / (2* self.sigma_jh_sq),
            axis=[-1, -2],
            keepdims=True
        )
        if self.verbose:
            print("p_j is:")    
            print(tf.exp(log_p_j))

        # Add log activations
        log_a_j = tf.math.log(a_j + epsilon)
        log_a_j = tf.broadcast_to(log_a_j, tf.shape(log_p_j))

        # Compute log numerator
        log_numerator = log_a_j + log_p_j

        # Compute log denominator using log-sum-exp for numerical stability
        log_denominator = tf.reduce_logsumexp(log_numerator, axis=[3, 4, 6], keepdims=True)

        # Compute log assignment probabilities
        log_R_ij = log_numerator - log_denominator

        # Clip log(R_ij), so that exponentiating it never leads to too large values
        # log_R_ij = tf.clip_by_value(log_R_ij, -200, 200)  # e^500 (=1.4e217) easily falls within the range of representable numbers by float64 (which is 1e300 or something)
        # Convert back from log-space
        self.R_ij = tf.exp(log_R_ij)
        if self.verbose:
            print("NEW R_ij is:")
            print(self.R_ij)

class Squeeze(tf.keras.layers.Layer):
    """ This layer only squeezes out the some singleton dimensions st the
    output suits the loss function better and the poses look nice
    """
    def __init__(self, **kwargs):
        super(Squeeze, self).__init__(**kwargs)

    def call(self, inputs):
        # Shapes are [batch, height, width, capsules, atom, atom]
        poses, acts = inputs
        
        acts = tf.debugging.check_numerics(acts, message="acts in Squeeze")
        poses = tf.debugging.check_numerics(poses, message="poses in Squeeze")
        # Squeeze unnecessary dimension out:
        # height, width are 1 at this point
        poses = tf.squeeze(poses, [1, 2])
        # poses is now [batch, capsules, atom, atom]

        acts = tf.squeeze(acts, [1, 2, 4, 5])
        # acts is now [batch, capsules] ie one-hot predictions

        return poses, acts


class DebugLayer(tf.keras.layers.Layer):
    """Layer to check intermediate values for bugs
    
    Tensorflow function can only be applied within a layer. In order to check 
    intermediate values for NaN or other bugs we have to pass them through a layer.
    """
    def __init__(self, msg="Check failed", **kwargs):
        super(DebugLayer, self).__init__(**kwargs)
        self.msg = msg


    def call(self, inputs):
        # if isinstance(inputs, (list, tuple)):
        #     for i, x in enumerate(inputs):
        #         tf.debugging.check_numerics(x, f"{self.msg} input {i}")
        #         if i ==1:
        #             print("In Layer:")
        #             print(self.msg)
        #             print("This are the activations:")
        #             print(x)
        # else:
        #     tf.debugging.check_numerics(inputs, self.msg)

        return inputs