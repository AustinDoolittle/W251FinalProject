import argparse
import logging
import sys

import tensorflow as tf
import numpy as np
import cv2

from constants import *
import decode


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

def process_frame(source_img):
    """https://github.com/rwightman/posenet-python/blob/97c6f6a3e0f25bbc9c9095ddef6e768e56c2ed43/posenet/utils.py#L13"""

    scale = np.array([source_img.shape[0] / IMAGE_SIZE, source_img.shape[1] / IMAGE_SIZE])

    input_img = cv2.resize(source_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = np.expand_dims(input_img, axis=0)

    return input_img, scale

def convert_raw_output_to_poses(pose_scores, keypoint_scores, keypoint_coords, scale, threshold=0.25):
    # TODO
    raise NotImplementedError()

def overlay_poses(poses, frame):
    # TODO
    raise NotImplementedError()

def publish_poses(poses):
    # TODO Wire up MQTT
    pass

def main(args):
    # Create our video capture device
    print('Opening video capture')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    print('successfully opened')
    
    with tf.Session() as sess:
        # load our model
        sess.graph.as_default()
        model_outputs = load_model(args.model_file, sess)

        # enter control loop
        c = 0
        while True:
            # read a frame
            res, frame = cap.read()
            c += 1

            # check if we were successful
            if not res:
                print(f'Failed to grab frame {c}')
                continue

            # scale down the frame, normalize the pixels
            input_image, scale = process_frame(frame)

            # run the image through our network
            heatmaps_out, offsets_out, displacement_fwd_out, displacement_bwd_out = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            # decode the output of the network 
            pose_scores, keypoint_scores, keypoint_coords = decode.decode_multiple_poses(
                heatmaps_out.squeeze(axis=0),
                offsets_out.squeeze(axis=0),
                displacement_fwd_out.squeeze(axis=0),
                displacement_bwd_out.squeeze(axis=0),
                output_stride=16,
                max_pose_detections=3,
                min_pose_score=0.25)

            # convert the decoded output to a nice json dictionary
            poses = convert_raw_output_to_poses(pose_scores, keypoint_scores, keypoint_coords, scale)

            # superimpose the wireframe on our grabbed image
            overlay_poses(poses, frame)

            # sent our poses to the MQTT topic
            publish_poses(poses)

            # show the people what we did
            cv2.imshow("person!", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
    