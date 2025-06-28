
import numpy as np
if __name__ != '__main__':
    # Didnt need these for some small tests (okay really ugly I know will remove this some time)
    import tensorflow as tf
    from utils.layers_em_hinton import ReLUConv, PrimaryCaps, ConvCaps, ClassCaps, EMRouting, DebugLayer, Squeeze

def em_capsnet_graph(input_shape, mode):
    """ Architecture of EM CapsNet, as described in: 'Matrix Capsules with EM Routing '

    Each layer is named after what their output represents
    """
    # Create position grid
    # input shape = [48, 48, 2]
    height, width = input_shape[0], input_shape[1]
    x = np.linspace(-1, 1, height)
    y = np.linspace(-1, 1, width)

    position_grid = np.meshgrid(x, y)


    inputs = tf.keras.Input(input_shape)
    # Other models in this repo need the y_true tensor for the reconstruction 
    # regularizer. We do not use y_true, but accept the input so that we can 
    # use the same Dataset object for training/testing
    y_true = tf.keras.layers.Input(shape=(5,))

    relu_conv1 = ReLUConv(A=32)(inputs)
    position_grid = position_grid_conv(position_grid, 5, 2, 'VALID') 

    prim_caps1 = PrimaryCaps()(relu_conv1)
    position_grid = position_grid_conv(position_grid, 1, 1, 'SAME')

    conv_caps1 = ConvCaps(stride=2)(prim_caps1)
    position_grid = position_grid_conv(position_grid, 3, 2, 'VALID')
    routing1 = EMRouting()(conv_caps1)    
  
    conv_caps2 = ConvCaps()(routing1)
    position_grid = position_grid_conv(position_grid, 3, 1, 'VALID')
    routing2 = EMRouting()(conv_caps2)

    class_caps = ClassCaps(position_grid)(routing2)

    outputs = EMRouting()(class_caps)

    poses, acts = outputs

    # TODO: if there are two inputs, return this model (commented out)
    # Makes sure we can use the same dataset as the other models, maybe better
    # to just add an error message or something
    # return tf.keras.Model(inputs=[inputs, y_true],outputs=prim_caps1, name='EM_CapsNet')

    return tf.keras.Model(inputs=inputs,outputs=outputs, name='EM_CapsNet')

def small_em_capsnet_graph(input_shape, mode, debug=True):
    # Create position grid
    # input shape = [48, 48, 2]
    height, width = input_shape[0], input_shape[1]
    x = np.linspace(-1, 1, height)
    y = np.linspace(-1, 1, width)

    position_grid = np.meshgrid(x, y)


    inputs = tf.keras.Input(input_shape)
    # Other models in this repo need the y_true tensor for the reconstruction 
    # regularizer. We do not use y_true, but accept the input so that we can 
    # use the same Dataset object for training/testing
    y_true = tf.keras.layers.Input(shape=(5,))

    relu_conv1 = ReLUConv(A=64)(inputs)
    # relu_conv1 = DebugLayer("relu_conv")(relu_conv1)
    position_grid = position_grid_conv(position_grid, 5, 2, 'VALID') 

    prim_caps1 = PrimaryCaps(B=8)(relu_conv1)
    # prim_caps1 = DebugLayer("primcaps")(prim_caps1)
    position_grid = position_grid_conv(position_grid, 1, 1, 'SAME')

    conv_caps1 = ConvCaps(C=16, stride=2)(prim_caps1)
    # conv_caps1 = DebugLayer("convcaps1")(conv_caps1)
    position_grid = position_grid_conv(position_grid, 3, 2, 'VALID')
    routing1 = EMRouting(name='routing1')(conv_caps1) 
    # routing1 = DebugLayer("routing1")(routing1) 
  
    conv_caps2 = ConvCaps(C=16)(routing1)
    # conv_caps2 = DebugLayer("convcaps2")(conv_caps2)
    position_grid = position_grid_conv(position_grid, 3, 1, 'VALID')
    routing2 = EMRouting()(conv_caps2) 
    # routing2 = DebugLayer("routing2")(routing2) 


    # class_caps = ClassCaps(position_grid)(routing2)
    class_caps = ConvCaps(C=5, kernel_size=8)(routing2)
    # class_caps = DebugLayer("classcaps")(class_caps)

    outputs = EMRouting()(class_caps) 
    # outputs = DebugLayer("routing_classcaps")(outputs)

    outputs = Squeeze()(outputs)

    poses, acts = outputs

    return tf.keras.Model(inputs=inputs,outputs=acts, name='small_EM_CapsNet')
   
def position_grid(grid, kernel_size, stride, padding):
    # Grid should be (1, H, W, 2) - where the 
    # last dimension is the x and y coordinates
    # Kernel should thus look like (kernel, kernel, 2, 2)
    #   So two outputs: an x and a y

    # TODO: Make a manual function that does this better. This gives issues 
    # due to padding, and really it is not that hard to calculate this myself
    x_kernel = tf.stack([tf.ones((kernel_size, kernel_size), dtype=tf.float32),
                         tf.zeros((kernel_size, kernel_size), dtype=tf.float32)],
                        axis=-1)
    y_kernel = tf.stack([tf.zeros((kernel_size, kernel_size), dtype=tf.float32),
                         tf.ones((kernel_size, kernel_size), dtype=tf.float32)],
                        axis=-1)
    
    kernel = tf.stack([x_kernel, y_kernel], axis=-1)
    strides = stride
    conv_position = tf.nn.conv2d(grid, kernel, strides, padding=padding)
    
    return conv_position / (kernel_size*kernel_size)

def position_grid_conv(grid, kernel_size, stride, padding):
    # grid is a tuple (xv, yv) that represents a mesh grid
    # To get the value at a position do:
    # x, y = grid[0][coordx, coordy], grid[1][coordx, coordy]
    # not super intuitive but oh well

    # We work with numpy, but output must be a list so that is serializable
    
    # Note that there is a stereo image input. We track just one since they
    # are treated exactly the same
    xv, yv = grid
    xv = np.array(xv)
    yv = np.array(yv)

    # Since everything is using tf, check that this is a np array
    assert(isinstance(xv, np.ndarray))


    if padding == 'SAME':
        xv = xv[::stride, ::stride]
        yv = yv[::stride, ::stride]
    elif padding == 'VALID':
        # Add 1 in case the kernel is even (assuming the center of the kernel
        # is a bit to the right in those cases)
        xv = xv[kernel_size//2:-(kernel_size-1)//2:stride, 
                kernel_size//2:-(kernel_size-1)//2:stride]
        yv = yv[kernel_size//2:-(kernel_size-1)//2:stride, 
                kernel_size//2:-(kernel_size-1)//2:stride]
    else:
        Exception("You passed an invalid padding value. Use 'SAME' or 'VALID'")
   
    # TODO: might be nice to track the size of the receptive field for later
    return (xv.tolist(), yv.tolist())

if __name__ == '__main__':
    x = np.arange(-16, 16)
    y = np.arange(-8, 8)
    grid = np.meshgrid(y, x)

    grid = position_grid_conv(grid, 5, 1, 'SAME')

    print(grid)


