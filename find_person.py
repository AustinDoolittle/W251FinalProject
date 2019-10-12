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
    parser.add_argument('--max-poses', default=2)
    parser.add_argument('--pose-threshold', type=float, default=0.5)
    parser.add_argument('--part-threshold', type=float, default=0.5)

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

def convert_raw_output_to_poses(pose_scores, keypoint_scores, keypoint_coords, scale):
    """Converts model output to a easy to understand dictionary

    Outputs a list of dictionaries, each dictionary follows this format:

    {
        "score": 0.75   # The overall score of the pose instance
        "parts": {      # Dictionary of the different body parts. See constants.PART_NAMES
            "nose": {                   # A single part instance   
                "score": 0.75,          # the score of this body part
                "coords": [123, 456]    # the coordinate of the body part, scaled to the original image size
            },
            ...
        }
    }

    Note that no thresholding occurs here, all instances and body parts are included in the output
    
    """

    converted_poses = []

    # iterate over instances
    for score, k_scores, k_coords in zip(pose_scores, keypoint_scores, keypoint_coords):

        # iterate over keypoints in this instance
        pose_dict = {
            'score': score,
            'parts': {}
        }
        for k_score, k_coord, part_name in zip(k_scores, k_coords, PART_NAMES):
            scaled_coord = (k_coord[0] * scale[0], k_coord[1] * scale[1])
            pose_dict['parts'][part_name] = {
                'coords': scaled_coord,
                'score': score
            }

        converted_poses.append(pose_dict)
    
    return converted_poses

def get_keypoints(parts, threshold=0.5):
    keypoints = []
    for part in parts:
        coords = parts[part]['coords']
        score = parts[part]['score']

        if score < threshold:
            continue

        keypoints.append(cv2.KeyPoint(coords[1], coords[0], 10 * score))
    return keypoints
    
def get_lines(parts, threshold=0.5):
    lines = []
    for left, right in CONNECTED_PART_NAMES:
        if parts[left]['score'] < threshold or parts[right]['score'] < threshold:
            continue

        lines.append(
            np.array([parts[left]['coords'][::-1], parts[right]['coords'][::-1]]).astype(np.int32),
        )
    return lines

def overlay_score(pose, frame, offset = 150):
    parts = pose['parts']
    pose_score = pose['score']

    # we'll just use the nose
    if not 'nose' in parts:
        return frame
    
    x_coord = int(parts['nose']['coords'][1])
    y_coord = int(parts['nose']['coords'][0]) - 50
    text = f'Score: {pose_score:.3f}'
    return cv2.putText(frame, text, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

def overlay_poses(poses, frame, instance_score_threshold=0.5, part_score_threshold=0.5):
    keypoints = []
    lines = []
    for pose in poses:
        if pose['score'] < part_score_threshold:
            continue
        
        parts = pose['parts']
        overlay_score(pose, frame)

        new_keypoints = get_keypoints(parts, threshold=part_score_threshold)
        keypoints += new_keypoints

        new_lines = get_lines(parts, threshold=part_score_threshold)
        lines += new_lines

    kp_frame = cv2.drawKeypoints(frame, keypoints, outImage=np.array([]))
    return cv2.polylines(kp_frame, lines, isClosed=False, color=(255,255,0))

def overlay_fps(cap, frame):
    # TODO this currently just does the stock fps
    # we'll need to determine a method for true fps 
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cv2.putText(frame, f'FPS: {fps:.3f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX,  
                       0.5, (0, 255, 0), 2, cv2.LINE_AA) 

def publish_poses(poses):
    # TODO Wire up MQTT
    pass

def main(args):
    # Create our video capture device
    print('Opening video capture')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
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
                max_pose_detections=args.max_poses,
                min_pose_score=args.pose_threshold)

            # convert the decoded output to a nice json dictionary
            poses = convert_raw_output_to_poses(pose_scores, keypoint_scores, keypoint_coords, scale)

            # superimpose the wireframe on our grabbed image
            out_frame = overlay_poses(
                poses, 
                frame, 
                instance_score_threshold=args.pose_threshold, 
                part_score_threshold=args.part_threshold
            )

            out_frame = overlay_fps(cap, out_frame)

            # sent our poses to the MQTT topic
            publish_poses(poses)

            # show the people what we did
            cv2.imshow("person!", out_frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
    