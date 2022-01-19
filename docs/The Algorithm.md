### ConvTasNet


```python
convtasnet.algorithm.ConvTasNet(
    num_sources=2,
    enc_kernel_size=16,
    enc_num_feats=512,
    msk_kernel_size=3,
    msk_num_feats=128,
    msk_num_hidden_feats=512,
    msk_num_layers=8,
    msk_num_stacks=3,
    msk_activate="sigmoid",
)
```


Separates the signals.

__Arguments__

- __num_sources (int, optional)__: The number of sources to split.
- __enc_kernel_size (int, optional)__: The convolution kernel size of the encoder/decoder.
- __enc_num_feats (int, optional)__: The feature dimensions passed to mask generator.
- __msk_kernel_size (int, optional)__: The convolution kernel size of the mask generator.
- __msk_num_feats (int, optional)__: The input/output feature dimension of conv block in the mask generator.
- __msk_num_hidden_feats (int, optional)__: The internal feature dimension of conv block of the mask generator.
- __msk_num_layers (int, optional)__: The number of layers in one conv block of the mask generator.
- __msk_num_stacks (int, optional)__: The numbr of conv blocks of the mask generator.
- __msk_activate (str, optional)__: The activation function of the mask output (Default: ``sigmoid``).

__Note__

This implementation corresponds to the "non-causal" setting in the paper.


----

### MaskGenerator


```python
convtasnet.algorithm.MaskGenerator(
    input_dim, num_sources, kernel_size, num_feats, num_hidden, num_layers, num_stacks, msk_activate
)
```


TCN (Temporal Convolution Network) Separation Module

Generates masks for separation.

__Arguments__

- __input_dim (int)__: Input feature dimension.
- __num_sources (int)__: The number of sources to separate.
- __kernel_size (int)__: The convolution kernel size of conv blocks.
- __num_feats (int)__: Input/output feature dimenstion of conv blocks.
- __num_hidden (int)__: Intermediate feature dimention of conv blocks.
- __num_layers (int)__: The number of conv blocks in one stack.
- __num_stacks (int)__: The number of conv block stacks.
- __msk_activate (str)__: The activation function of the mask output.


----

### ConvBlock


```python
convtasnet.algorithm.ConvBlock(filters, kernel_size, no_residual=False)
```


1D Convolutional block.

__Arguments__

- __filters (int)__: The number of channels in the internal layers.
- __kernel_size (int)__: The convolution kernel size of the middle layer.
- __no_redisual (bool, optional)__: Disable residual block/output.


----

