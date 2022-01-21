import tensorflow as tf
from tensorflow.keras import Model, layers


class ConvBlock(layers.Layer):
    """1D Convolutional block.

    # Arguments

        filters (int): The number of channels in the internal layers.
        kernel_size (int): The convolution kernel size of the middle layer.
        no_residual (bool, optional): Disable residual block/output.

    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        no_residual: bool = False,
    ):

        super(ConvBlock, self).__init__()

        self.conv_layers = tf.keras.Sequential(
            [
                layers.Conv1D(filters, kernel_size),
                layers.PReLU(),
                layers.LayerNormalization(),
                layers.Conv1D(filters, kernel_size),
                layers.PReLU(),
                layers.LayerNormalization(),
            ]
        )

        self.res_out = None if no_residual else layers.Conv1D(filters, kernel_size)

        self.skip_out = layers.Conv1D(filters, kernel_size)

    def call(self, input: tf.Tensor):

        feature = self.conv_layers(input)

        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)

        skip_out = self.skip_out(feature)

        return residual, skip_out


class MaskGenerator(layers.Layer):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    # Arguments

        input_dim (int): Input feature dimension.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks.
        num_feats (int): Input/output feature dimenstion of conv blocks.
        num_hidden (int): Intermediate feature dimention of conv blocks.
        num_layers (int): The number of conv blocks in one stack.
        num_stacks (int): The number of conv block stacks.
        msk_activate (str): The activation function of the mask output.

    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
    ):

        super(MaskGenerator, self).__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = layers.LayerNormalization()
        self.input_conv = layers.Conv1D(num_feats, kernel_size)

        self.receptive_field = 0

        self.conv_layers = []
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(
                    ConvBlock(
                        num_feats,
                        kernel_size,
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += (
                    kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
                )

        self.output_prelu = layers.PReLU()
        self.output_conv = layers.Conv1D(num_feats, kernel_size)

        if msk_activate == "sigmoid":
            self.mask_activate = tf.keras.activations.sigmoid
        elif msk_activate == "relu":
            self.mask_activate = tf.keras.activations.relu

    def call(self, input: tf.Tensor):

        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0

        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:
                feats += residual
            output += skip

        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)

        return tf.reshape(output, (batch_size, self.num_sources, self.input_dim))


class ConvTasNet(Model):
    """Separates the signals.

    # Arguments

        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).

    # Note

        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        num_sources: int = 2,
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
    ):

        super(ConvTasNet, self).__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = tf.keras.layers.Conv1D(enc_num_feats, enc_kernel_size)

        self.mask_generator = MaskGenerator(
            enc_num_feats,
            num_sources,
            msk_kernel_size,
            msk_num_feats,
            msk_num_hidden_feats,
            msk_num_layers,
            msk_num_stacks,
            msk_activate,
        )

        self.decoder = tf.keras.layers.Conv1DTranspose(1, enc_kernel_size)

    def pad(self, input: tf.Tensor):

        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)

        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = tf.zeros(
            shape=(batch_size, num_channels, num_paddings), dtype=input.dtype
        )

        return tf.concat([input, pad], 2)

    def call(self, input: tf.Tensor):

        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channels==1, frame). Found {input.shape}"
            )

        padded, num_pads = self.pad(input)
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)
        masked = self.mask_generator(feats) * tf.expand_dims(input, 1)
        masked = tf.reshape(input, (batch_size, self.num_sources, num_padded_frames))

        decoded = self.decoder(masked)
        output = tf.reshape(decoded, (batch_size, self.num_sources, num_padded_frames))

        if num_pads > 0:
            output = output[..., :-num_pads]

        return output