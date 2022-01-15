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
- __enc_kernel_size (int, optional)__: The convolution kernel size of the encoder/decoder, <L>.
- __enc_num_feats (int, optional)__: The feature dimensions passed to mask generator, <N>.
- __msk_kernel_size (int, optional)__: The convolution kernel size of the mask generator, <P>.
- __msk_num_feats (int, optional)__: The input/output feature dimension of conv block in the mask generator, <B, Sc>.
- __msk_num_hidden_feats (int, optional)__: The internal feature dimension of conv block of the mask generator, <H>.
- __msk_num_layers (int, optional)__: The number of layers in one conv block of the mask generator, <X>.
- __msk_num_stacks (int, optional)__: The numbr of conv blocks of the mask generator, <R>.
- __msk_activate (str, optional)__: The activation function of the mask output (Default: ``sigmoid``).

__Note__

This implementation corresponds to the "non-causal" setting in the paper.


----

