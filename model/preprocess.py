import os
import pandas as pd
import tensorflow as tf
"""
example: https://www.tensorflow.org/tutorials/audio/simple_audio#build_and_train_the_model
"""


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


if __name__ == '__main__':

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

        for mix in paths["mixture_path"]:
            wavf = tf.io.read_file(mix)
            audio, _ = tf.audio.decode_wav(contents=wavf)
            audio = tf.squeeze(audio, axis=-1)