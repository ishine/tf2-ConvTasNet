import os
import pandas as pd
import tensorflow as tf

if os.getcwd() != 'tf2-ConvTasNet':
    os.chdir('..')

for dataset in ['train', 'val']:
    paths = pd.read_csv(
        f"MiniLibriMix/metadata/mixture_{dataset}_mix_clean.csv")
    ds = pd.DataFrame([
        tf.audio.decode_wav(contents=tf.io.read_file(mix))
        for mix in paths["mixture_path"]
    ])

    signals = []

    for i in range(ds.shape[0]):
        signals.append([ds['audio'].iloc[i].numpy().flatten()])
    audio = pd.DataFrame(signals, columns=['signal'])
    name = 'test' if dataset == 'val' else dataset
    audio.to_parquet(f'data/{name}.parquet')