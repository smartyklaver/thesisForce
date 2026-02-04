import os
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model


def build_and_save(path):
    # Input shape matches recorded sample: (channels, time, 1) -> (8,100,1)
    inp = Input(shape=(8, 100, 1), name='input')
    x = Conv2D(16, (1, 5), activation='relu', padding='same')(inp)
    x = MaxPooling2D((1, 2))(x)
    x = Conv2D(32, (1, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='dense_1')(x)
    # This is the layer `fine_tune.py` expects to extract features from
    x = Dense(64, activation='relu', name='dense_8')(x)
    out = Dense(4, activation='softmax', name='output')(x)

    model = Model(inputs=inp, outputs=out)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Saved dummy feature-extractor to: {path}")


if __name__ == '__main__':
    target = os.path.join(os.path.dirname(__file__), 'og_fine_tune.h5')
    build_and_save(target)
