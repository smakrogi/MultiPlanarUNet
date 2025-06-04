from keras_unet_collection import models
from keras import activations
from keras.models import Model
from keras.layers import Reshape
from mpunet.logging import ScreenLogger
import numpy as np


class Trans_UNet_2D(Model):
    """
    2D Trans UNet implementation. 
    """

    def __init__(self,
                 n_classes,
                 img_rows=None,
                 img_cols=None,
                 dim=None,
                 n_channels=1,
                 depth=4,
                 out_activation="softmax",
                 activation="relu",
                 kernel_size=3,
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=None,
                 logger=None,
                 **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        img_rows, img_cols (int, int):
            Image dimensions. Note that depending on image dims cropping may
            be necessary. To avoid this, use image dimensions DxD for which
            D * (1/2)^n is an integer, where n is the number of (2x2)
            max-pooling layers; in this implementation 4.
            For n=4, D \in {..., 192, 208, 224, 240, 256, ...} etc.
        dim (int):
            img_rows and img_cols will both be set to 'dim'
        n_channels (int):
            Number of channels in the input image.
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        out_activation (string):
            Activation function of output 1x1 conv layer. Usually one of
            'softmax', 'sigmoid' or 'linear'.
        activation (string):
            Activation function for convolution layers
        kernel_size (int):
            Kernel size for convolution layers
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            2D convolution layer instead of default N.
        flatten_output (bool):
            Flatten the output to array of shape [batch_size, -1, n_classes]
        l2_reg (float in [0, 1])
            L2 regularization on Conv2D weights
        logger (mpunet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        """
        super(Trans_UNet_2D, self).__init__()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim
        # breakpoint()
            
        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        # Set various attributes
        self.cf = np.sqrt(complexity_factor)
        self.kernel_size = kernel_size
        self.activation = activation
        self.out_activation = out_activation
        self.l2_reg = l2_reg
        self.padding = padding
        self.depth = depth
        self.flatten_output = flatten_output

        self.n_classes = n_classes
        self.img_shape = (img_rows, img_cols, n_channels)
        self.out_activation = out_activation
        
        self.num_layers = 12
        self.hidden_dim = 64
        self.mlp_dim = 128
        self.num_heads = 6 
        self.dropout_rate = 0.1
        self.patch_size = 16
        self.num_patches = (self.img_shape[0]*self.img_shape[1])//(self.patch_size**2)
        self.num_channels = n_channels

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        # names = [x.__class__.__name__ for x in self.layers]
        # index = names.index("Conv2DTranspose")
        # self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()



    def init_model(self, 
                   data_augmentation = None,
                   num_filters = 16, 
                   decoder_activation = 'relu',
                   decoder_kernel_init = 'he_normal',
                   ViT_hidd_mult = 3,
                   batch_norm = True,
                   dropout = 0.0
                   ):

        
        model = models.transunet_2d(self.img_shape, filter_num=[64, 128, 256, 512], n_labels=12, 
                                    stack_num_down=2, stack_num_up=2,
                                    embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                                    activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
                                    batch_norm=True, pool=True, unpool='bilinear', name='transunet')

        # breakpoint()
        # model.layers[-2].activation = activations.softmax
        # model.compile()
        inputs = model.input
        output = model.output

        if self.flatten_output:
            output = Reshape([self.img_shape[0]*self.img_shape[1],
                           self.n_classes], name='flatten_output')(output)

        return [inputs], [output]


    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        # self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
