import tensorflow as tf
from mltu.annotations.images import CVImage
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric

import os
from tqdm import tqdm
from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block

import pandas as pd
from mltu.configs import BaseModelConfigs
import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

import yaml

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):

    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x8 = residual_block(x7, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x9 = residual_block(x8, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    # blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    # blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model

def get_config(config_path = "weights/configs.yaml"):
    with open('weights/configs.yaml') as f:
        my_config = yaml.safe_load(f)
    
    return my_config

def get_model(configs, weight_path = "weights/model.h5"):    
    model = train_model(input_dim = (configs['height'], configs['width'], 3),
                    output_dim = len(configs['vocab']))
    
    model.load_weights(weight_path)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = configs['learning_rate']),
        loss = CTCloss(),
        metrics = [
            CERMetric(vocabulary = configs['vocab']),
            WERMetric(vocabulary = configs['vocab'])
        ],
        run_eagerly = False)
    # model.summary()
    return model

def predict(image):
        configs = get_config()
        model = get_model(configs)

        image = ImageResizer.resize_maintaining_aspect_ratio(image, 1408, 96)

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = model.predict(image_pred)

        text = ctc_decoder(preds, configs['vocab'])[0]

        return text
