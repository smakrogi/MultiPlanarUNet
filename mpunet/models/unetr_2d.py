"""
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from math import log2
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model

from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape
import numpy as np

class Patches(L.Layer):
    # It takes a batch of images and returns a batch of patches
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class UNetR_2D(Model):
    """
    2D UNetR implementation. 
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
        super(UNetR_2D, self).__init__()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

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

    def mlp(self, x):
        x = L.Dense(self.mlp_dim, activation="gelu")(x)
        x = L.Dropout(self.dropout_rate)(x)
        x = L.Dense(self.hidden_dim)(x)
        x = L.Dropout(self.dropout_rate)(x)
        return x

    def transformer_encoder(self, x):
        skip_1 = x
        x = L.LayerNormalization()(x)
        x = L.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim
        )(x, x)
        x = L.Add()([x, skip_1])

        skip_2 = x
        x = L.LayerNormalization()(x)
        x = self.mlp(x)
        x = L.Add()([x, skip_2])

        return x

    def conv_block(self, x, num_filters, kernel_size=3):
        x = L.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        return x

    def deconv_block(self, x, num_filters, strides=2):
        x = L.Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=strides)(x)
        return x

    def init_model(self):
        """ Inputs """
        inputs = L.Input(self.img_shape) 
        patches = Patches(self.patch_size)(inputs)
        
        patch_shape = (self.num_patches, self.patch_size*self.patch_size*self.num_channels)

        """ Patch + Position Embeddings """
        patch_embed = L.Dense(self.hidden_dim)(patches) ## (None, 256, 768)

        positions = tf.range(start=0, limit=self.num_patches, delta=1) ## (256,)
        pos_embed = L.Embedding(input_dim=self.num_patches, output_dim=self.hidden_dim)(positions) ## (256, 768)
        x = patch_embed + pos_embed ## (None, 256, 768)

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for i in range(1, self.num_layers+1, 1):
            x = self.transformer_encoder(x)

            if i in skip_connection_index:
                skip_connections.append(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        ## Reshaping
        z0 = L.Reshape((self.img_shape[0], self.img_shape[1], self.num_channels))(patches)

        shape = (
            self.img_shape[0]//self.patch_size,
            self.img_shape[1]//self.patch_size,
            self.hidden_dim
        )
        z3 = L.Reshape(shape)(z3)
        z6 = L.Reshape(shape)(z6)
        z9 = L.Reshape(shape)(z9)
        z12 = L.Reshape(shape)(z12)

        ## Additional layers for managing different patch sizes
        total_upscale_factor = int(log2(self.patch_size))
        upscale = total_upscale_factor - 4

        if upscale >= 1: ## Patch size 16 or greater
            z3 = self.deconv_block(z3, z3.shape[-1], strides=2**upscale)
            z6 = self.deconv_block(z6, z6.shape[-1], strides=2**upscale)
            z9 = self.deconv_block(z9, z9.shape[-1], strides=2**upscale)
            z12 = self.deconv_block(z12, z12.shape[-1], strides=2**upscale)
            # print(z3.shape, z6.shape, z9.shape, z12.shape)

        if upscale < 0: ## Patch size less than 16
            p = 2**abs(upscale)
            z3 = L.MaxPool2D((p, p))(z3)
            z6 = L.MaxPool2D((p, p))(z6)
            z9 = L.MaxPool2D((p, p))(z9)
            z12 = L.MaxPool2D((p, p))(z12)

        ## Decoder 1
        x = self.deconv_block(z12, 128)
        s = self.deconv_block(z9, 128)
        s = self.conv_block(s, 128)

        x = L.Concatenate()([x, s])

        x = self.conv_block(x, 128)
        x = self.conv_block(x, 128)

        ## Decoder 2
        x = self.deconv_block(x, 64)

        s = self.deconv_block(z6, 64)
        s = self.conv_block(s, 64)
        s = self.deconv_block(s, 64)
        s = self.conv_block(s, 64)

        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 64)
        x = self.conv_block(x, 64)

        ## Decoder 3
        x = self.deconv_block(x, 32)

        s = self.deconv_block(z3, 32)
        s = self.conv_block(s, 32)
        s = self.deconv_block(s, 32)
        s = self.conv_block(s, 32)
        s = self.deconv_block(s, 32)
        s = self.conv_block(s, 32)

        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 32)
        x = self.conv_block(x, 32)

        ## Decoder 4
        x = self.deconv_block(x, 16)

        s = self.conv_block(z0, 16)
        s = self.conv_block(s, 16)

        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 16)
        x = self.conv_block(x, 16)

        """ Output """
        out = L.Conv2D(self.n_classes, kernel_size=1, padding="same", activation=self.out_activation)(x)

        if self.flatten_output:
            out = Reshape([self.img_shape[0]*self.img_shape[1],
                           self.n_classes], name='flatten_output')(out)

        return [inputs], [out]

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
