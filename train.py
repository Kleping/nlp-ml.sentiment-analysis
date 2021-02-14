import numpy as np
import tensorflow as tf
import random as rn
import re
import json


validation_split = .2

epochs = 100
batch_size = 128
LATENT_DIM = 32

model_name = 'model.h5'

data_path = 'data/input.txt'
model_path = 'models/model.h5'

punctuation = ['!', '?', '.', ',']
sentinels = ['<BOS>', '<EOS>']
OOV_TOKEN = '<OOV>'
PAD_TOKEN = '<PAD>'


def split_with_keep_delimiters(text, delimiters):
    return [
        token for token in re.split('(' + '|'.join(map(re.escape, delimiters)) + ')', text) if token is not ''
    ]


def tokenize_text(text):
    tokens = list()
    for token in text.split():
        if any(map(str.isdigit, token)):
            if token[-1] in punctuation:
                tokens.append(token[:-1])
                tokens.append(token[-1])
            else:
                tokens.append(token)
        else:
            [tokens.append(split_token) for split_token in split_with_keep_delimiters(token, punctuation)]

    return tokens


def find_max_input_len(_x):
    _len = 0
    for i in _x:
        tokens = tokenize_text(i)
        found_len = len(tokens)
        if found_len > _len:
            _len = found_len

    return _len


def split_data(_x, _y):
    _x_valid = _x[-int(validation_split * len(_x)):]
    _x_train = _x[:int(len(_x) - len(_x_valid))]

    _y_valid = _y[-int(validation_split * len(_y)):]
    _y_train = _y[:int(len(_y) - len(_y_valid))]

    _x_residual = len(_x_valid) // 2
    _x = (_x_train, _x_valid[-_x_residual:], _x_valid[:_x_residual])

    _y_residual = len(_y_valid) // 2
    _y = (_y_train, _y_valid[-_y_residual:], _y_valid[:_y_residual])

    return _x, _y


def create_vocabulary(_x):
    # a bag of words
    bow = list()
    for i in _x:
        [bow.append(token) for token in tokenize_text(i) if token not in bow]

    _voc = (bow + [OOV_TOKEN])
    rn.shuffle(_voc)
    return [PAD_TOKEN] + _voc


def serialize_and_write_config(_voc, _max_input_len, _history):
    config = {
        'voc': _voc,
        'max_input_len': _max_input_len,
        'history': _history
    }

    with open('models/model.config', 'w') as config_file:
        json.dump(config, config_file, indent=4)


# read all the data
with open('data/input.txt', 'r', encoding='utf-8') as f:
    x = list(filter(None, f.read().split('\n')))

with open('data/output.txt', 'r', encoding='utf-8') as f:
    y = list(filter(None, f.read().split('\n')))


# shuffle lists
c = list(zip(x, y))
rn.shuffle(c)
x, y = zip(*c)


(x_train, x_valid, x_test), (y_train, y_valid, y_test) = split_data(x, y)

print("amount of training data: " + str(len(x_train) + len(x_valid)))
print("amount of test data: " + str(len(x_test)))

max_input_len = max(find_max_input_len(x_train), find_max_input_len(x_valid))
voc = create_vocabulary(x_train)


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, _batch, _max_input_len, _x, _y, _voc):
        self._batch = _batch
        self._x = _x
        self._y = _y
        self._voc = _voc
        self._max_input_len = _max_input_len

    def __len__(self):
        return int(np.floor(len(self._x) / self._batch))

    def __getitem__(self, _batch_index):
        _x, _y = self.extract_batch(_batch_index)
        return self.encode(_x, _y)

    def on_epoch_end(self):
        _c = list(zip(self._x, self._y))
        rn.shuffle(_c)
        self._x, self._y = zip(*_c)

    # secondary auxiliary methods
    def find_index(self, token):
        if token in self._voc:
            _i = self._voc.index(token)
        else:
            _i = self._voc.index(OOV_TOKEN)
        return _i

    def encode(self, _in_x, _in_y):
        _x = np.zeros((len(_in_x), self._max_input_len), dtype='int32')
        _y = np.zeros((len(_in_y), 10), dtype='float32')

        for raw_index, (_x_item, _y_item) in enumerate(zip(_in_x, _in_y)):
            for order_index, token in enumerate(tokenize_text(_x_item)):
                _x[raw_index, order_index] = self.find_index(token)

            _y[raw_index, int(_in_y[raw_index]) - 1] = 1.

        return _x, _y

    def extract_batch(self, _batch_index):
        _index_from = _batch_index * self._batch
        _index_to = min(_batch_index * self._batch + self._batch, len(self._x))

        _x = self._x[_index_from: _index_to]
        _y = self._y[_index_from: _index_to]

        if len(_x) < self._batch:
            _c = list(zip(self._x, self._y))
            _x_additional, _y_additional = zip(*rn.sample(_c[:_index_from], self._batch - len(_x)))
            _x = _x + _x_additional
            _y = _y + _y_additional

        return _x, _y


# architecture model creation
lstm_cell = tf.keras.layers.LSTM(LATENT_DIM, return_sequences=True, recurrent_dropout=.1)
lstm_cell_secondary = tf.keras.layers.LSTM(LATENT_DIM*2, return_sequences=True, recurrent_dropout=.1)

model = tf.keras.Sequential()
model . add(tf.keras.Input(shape=(None,)))
model . add(tf.keras.layers.Embedding(len(voc), LATENT_DIM))
model . add(tf.keras.layers.Bidirectional(lstm_cell))
model . add(tf.keras.layers.Bidirectional(lstm_cell_secondary))
model . add(tf.keras.layers.GlobalMaxPooling1D())
model . add(tf.keras.layers.Dense(10))
model . add(tf.keras.layers.Activation("softmax"))

model . summary()

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

train_supplier = DataSupplier(
    batch_size,
    max_input_len,
    x_train,
    y_train,
    voc
)

valid_supplier = DataSupplier(
    batch_size,
    max_input_len,
    x_valid,
    y_valid,
    voc
)

history = model.fit(
    train_supplier,
    validation_data=valid_supplier,
    epochs=epochs,
    shuffle=True,
).history

model.save(model_path)
serialize_and_write_config(voc, max_input_len, history)
