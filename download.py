"""
Slightly modified version of https://github.com/rwightman/posenet-python
"""

import argparse
import json
import tempfile
import os
import posixpath
import struct
import sys

import numpy as np
import requests
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

mobilenet100_architecture = [
    ('conv2d', 2),
    ('separableConv', 1),
    ('separableConv', 2),
    ('separableConv', 1),
    ('separableConv', 2),
    ('separableConv', 1),
    ('separableConv', 2),
    ('separableConv', 1),
    ('separableConv', 1),
    ('separableConv', 1),
    ('separableConv', 1),
    ('separableConv', 1),
    ('separableConv', 2),
    ('separableConv', 1),
]


MODEL_BASE_URL = "https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_101"
MANIFEST_FILENAME = 'manifest.json'
OUTPUT_STRIDE = 16
IMAGE_SIZE = 513


def _download_file(output_dir, filename):
    url = posixpath.join(MODEL_BASE_URL, filename)
    response = requests.get(url)
    response.raise_for_status()

    out_filename = os.path.join(output_dir, filename)
    with open(out_filename, 'wb') as fp:
        fp.write(response.content)

    return out_filename

    
def _download_model(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # retrieve the manifest
    manifest_path = _download_file(output_dir, MANIFEST_FILENAME)

    # load the manifest into memory
    with open(manifest_path, 'r') as fp:
        manifest = json.load(fp)

    # iterate over the items in the manifest and download the weights
    for item in manifest.values():
        filename = item['filename']
        print(f'Downloading {filename}... ', end='')
        _download_file(output_dir, filename)
        print('Done')


def to_output_strided_layers(convolution_def):
    current_stride = 1
    rate = 1
    block_id = 0
    buff = []
    for _a in convolution_def:
        conv_type = _a[0]
        stride = _a[1]
        
        if current_stride == OUTPUT_STRIDE:
            layer_stride = 1
            layer_rate = rate
            rate *= stride
        else:
            layer_stride = stride
            layer_rate = 1
            current_stride *= stride
        
        buff.append({
            'blockId': block_id,
            'convType': conv_type,
            'stride': layer_stride,
            'rate': layer_rate,
            'outputStride': current_stride
        })
        block_id += 1

    return buff


def load_variables(model_path):
    manifest_path = os.path.join(model_path, MANIFEST_FILENAME)
    with open(manifest_path) as f:
        variables = json.load(f)

    for x in variables:
        filename = variables[x]["filename"]
        full_filename = os.path.join(model_path, filename)
        layer_def = open(full_filename, 'rb').read()
        fmt = str(int(len(layer_def) / struct.calcsize('f'))) + 'f'
        d = struct.unpack(fmt, layer_def)
        d = tf.cast(d, tf.float32)
        d = tf.reshape(d, variables[x]["shape"])
        variables[x]["x"] = tf.Variable(d, name=x)

    return variables

def build_network(image, layers, variables):

    def _weights(layer_name):
        return variables["MobilenetV1/" + layer_name + "/weights"]['x']

    def _biases(layer_name):
        return variables["MobilenetV1/" + layer_name + "/biases"]['x']

    def _depthwise_weights(layer_name):
        return variables["MobilenetV1/" + layer_name + "/depthwise_weights"]['x']

    def _conv_to_output(mobile_net_output, output_layer_name):
        w = tf.nn.conv2d(mobile_net_output, _weights(output_layer_name), [1, 1, 1, 1], padding='SAME')
        w = tf.nn.bias_add(w, _biases(output_layer_name), name=output_layer_name)
        return w

    def _conv(inputs, stride, block_id):
        return tf.nn.relu6(
            tf.nn.conv2d(inputs, _weights("Conv2d_" + str(block_id)), stride, padding='SAME')
            + _biases("Conv2d_" + str(block_id)))

    def _separable_conv(inputs, stride, block_id, dilations):
        if dilations is None:
            dilations = [1, 1]

        dw_layer = "Conv2d_" + str(block_id) + "_depthwise"
        pw_layer = "Conv2d_" + str(block_id) + "_pointwise"

        w = tf.nn.depthwise_conv2d(
            inputs, _depthwise_weights(dw_layer), stride, 'SAME', rate=dilations, data_format='NHWC')
        w = tf.nn.bias_add(w, _biases(dw_layer))
        w = tf.nn.relu6(w)

        w = tf.nn.conv2d(w, _weights(pw_layer), [1, 1, 1, 1], padding='SAME')
        w = tf.nn.bias_add(w, _biases(pw_layer))
        w = tf.nn.relu6(w)

        return w

    x = image
    buff = []
    with tf.variable_scope(None, 'MobilenetV1'):

        for m in layers:
            stride = [1, m['stride'], m['stride'], 1]
            rate = [m['rate'], m['rate']]
            if m['convType'] == "conv2d":
                x = _conv(x, stride, m['blockId'])
                buff.append(x)
            elif m['convType'] == "separableConv":
                x = _separable_conv(x, stride, m['blockId'], rate)
                buff.append(x)

    heatmaps = _conv_to_output(x, 'heatmap_2')
    offsets = _conv_to_output(x, 'offset_2')
    displacement_fwd = _conv_to_output(x, 'displacement_fwd_2')
    displacement_bwd = _conv_to_output(x, 'displacement_bwd_2')
    heatmaps = tf.sigmoid(heatmaps, 'heatmap')

    return heatmaps, offsets, displacement_fwd, displacement_bwd


def convert(input_dir, output_model_file):
    os.makedirs(input_dir, exist_ok=True)

    output_dir = os.path.dirname(output_model_file)
    os.makedirs(output_dir, exist_ok=True)

    cg = tf.Graph()
    with cg.as_default():
        layers = to_output_strided_layers(mobilenet100_architecture)
        variables = load_variables(input_dir)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()

            image_ph = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image')
            outputs = build_network(image_ph, layers, variables)

            sess.run(
                [outputs],
                feed_dict={
                    image_ph: [np.ndarray(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)]
                }
            )
            
            ckpt_path = os.path.join(output_dir, 'model.ckpt')
            checkpoint_path = saver.save(sess, ckpt_path, write_state=False)

            tf.train.write_graph(cg, output_dir, 'model.pbtxt')

            # Freeze graph and write our final model file
            freeze_graph(
                input_graph=os.path.join(output_dir, "model.pbtxt"),
                input_saver="",
                input_binary=False,
                input_checkpoint=checkpoint_path,
                output_node_names='heatmap,offset_2,displacement_fwd_2,displacement_bwd_2',
                restore_op_name="save/restore_all",
                filename_tensor_name="save/Const:0",
                output_graph=output_model_file,
                clear_devices=True,
                initializer_nodes="")

    return output_model_file


def download_and_convert(output_model_file):
    with tempfile.TemporaryDirectory() as workdir_name:
        _download_model(workdir_name)

        convert(workdir_name, output_model_file)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-model-file', default='./out/model.pb')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    download_and_convert(args.output_model_file)