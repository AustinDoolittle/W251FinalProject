import argparse
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


def traverse_to_targ_keypoint(
        edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements
):
    """Borrowed from https://github.com/rwightman/posenet-python/blob/master/posenet/decode.py"""
    height = scores.shape[0]
    width = scores.shape[1]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    displaced_point = source_keypoint + displacements[
        source_keypoint_indices[0], source_keypoint_indices[1], edge_id]

    displaced_point_indices = np.clip(
        np.round(displaced_point / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    score = scores[displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    image_coord = displaced_point_indices * output_stride + offsets[
        displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    return score, image_coord


def decode_pose(
        root_score, root_id, root_image_coord,
        scores,
        offsets,
        output_stride,
        displacements_fwd,
        displacements_bwd
):
    """Borrowed from https://github.com/rwightman/posenet-python/blob/master/posenet/decode.py"""
    num_parts = scores.shape[2]
    num_edges = len(PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_bwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    return instance_keypoint_scores, instance_keypoint_coords


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
    # input_img = input_img.reshape(1, IMAGE_SIZE, target_width, 3)
    input_img = np.expand_dims(input_img, axis=0)

    return input_img, scale

def main(args):
    print('Opening video capture')
    cap = cv2.VideoCapture(0)
    print('successfully opened')
    
    with tf.Session() as sess:
        sess.graph.as_default()
        model_outputs = load_model(args.model_file, sess)

        c = 0
        while True:
            res, frame = cap.read()
            c += 1

            if not res:
                print('Failed to grab frame {c}')
                continue

            input_image, scale = process_frame(frame)

            heatmaps_out, offsets_out, displacement_fwd_out, displacement_bwd_out = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = decode.decode_multiple_poses(
                heatmaps_out.squeeze(axis=0),
                offsets_out.squeeze(axis=0),
                displacement_fwd_out.squeeze(axis=0),
                displacement_bwd_out.squeeze(axis=0),
                output_stride=16,
                max_pose_detections=3,
                min_pose_score=0.25)
            
            print(pose_scores, keypoint_scores, keypoint_coords)

            cv2.imshow("person!", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
    