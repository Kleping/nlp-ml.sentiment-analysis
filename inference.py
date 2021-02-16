import re
import numpy as np
import tensorflow as tf
import json

batch_size = 128
coefficient = 500
latent_dim = 32

model_name = 'model'

language_tag = 'en'
data_path = 'data/{}/paired.txt'.format(language_tag)


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


def encode_target(text):
    return '{} {} {}'.format(sentinels[0], text, sentinels[1])


def deserialize_and_read_config():
    with open('models/model.config', 'r') as config_file:
        config = json.load(config_file)

    return config['voc'], config['max_input_len']


def get_token_index(voc, token):
    if token in voc:
        voc_i = voc.index(token)
    else:
        voc_i = voc.index(OOV_TOKEN)
    return voc_i


def encode(_sentence, _voc, _max_input_len):

    encoded_text = np.zeros((1, _max_input_len), dtype='int32')
    for t, token in enumerate(tokenize_text(_sentence)):
        encoded_text[0, t] = get_token_index(_voc, token)
    return encoded_text


with open('data/input.txt', 'r', encoding='utf-8') as f:
    x = list(filter(None, f.read().split('\n')[1500:1600]))

with open('data/output.txt', 'r', encoding='utf-8') as f:
    y = list(filter(None, f.read().split('\n')[1500:1600]))

voc, max_input_len = deserialize_and_read_config()

model = tf.keras.models.load_model('models/model.h5')
print(model.summary())

counter = 0

for i in range(len(x)):
    sentence = x[i]
    correct_label = int(y[i])
    result = model.predict(encode(sentence, voc, max_input_len))[0]
    predicted_tonality = np.argmax(result) + 1
    if correct_label == predicted_tonality:
        counter += 1

print('{} out of {}'.format(counter, len(x)))