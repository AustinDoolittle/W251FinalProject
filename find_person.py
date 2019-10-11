import argparse
import sys

import tensorflow as tf
import cv2


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', required=True)

    return parser.parse_args(argv)

def load_model(model_file, sess):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fp:
        graph_def.ParseFromString(fp.read())
    
    tf.import_graph_def(graph_def, name='')

    offsets = sess.graph.get_tensor_by_name('offset_2:0')
    displacement_fwd = sess.graph.get_tensor_by_name('displacement_fwd_2:0')
    displacement_bwd = sess.graph.get_tensor_by_name('displacement_bwd_2:0')
    heatmaps = sess.graph.get_tensor_by_name('heatmap:0')

    return heatmaps, offsets, displacement_fwd, displacement_bwd

def main(args):
    print('Opening video capture')
    cap = cv2.VideoCapture(0)
    print('successfully opened')
    
    with tf.Session() as sess:
        sess.graph.as_default()
        heatmaps, offsets, displacement_fwd, displacement_bwd = load_model(args.model_file, sess)
        c = 0
        while True:
            res, frame = cap.read()
            c += 1

            if not res:
                print('Failed to grab frame {c}')
                continue

            cv2.imshow("person!", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
    