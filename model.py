import tensorflow as tf
import numpy as np
import cv2

from constants import *
import decode


class PoseNetModel:
    def __init__(self, model_file):
        # I got a CUDNN init error without this config
        # ¯\_(ツ)_/¯
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self._sess.as_default()

        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_file, 'rb') as fp:
            graph_def.ParseFromString(fp.read())
        
        tf.import_graph_def(graph_def, name='')

        self._offsets = self._sess.graph.get_tensor_by_name('offset_2:0')
        self._displacement_fwd = self._sess.graph.get_tensor_by_name('displacement_fwd_2:0')
        self._displacement_bwd = self._sess.graph.get_tensor_by_name('displacement_bwd_2:0')
        self._heatmaps = self._sess.graph.get_tensor_by_name('heatmap:0')

    def _process_frame(self, source_img):
        """https://github.com/rwightman/posenet-python/blob/97c6f6a3e0f25bbc9c9095ddef6e768e56c2ed43/posenet/utils.py#L13"""

        scale = np.array([source_img.shape[0] / IMAGE_SIZE, source_img.shape[1] / IMAGE_SIZE])

        input_img = cv2.resize(source_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_img = input_img * (2.0 / 255.0) - 1.0
        input_img = np.expand_dims(input_img, axis=0)

        return input_img, scale

    def _convert_raw_output_to_poses(self, pose_scores, keypoint_scores, keypoint_coords, scale):
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
                    'score': k_score
                }

            converted_poses.append(pose_dict)
        
        return converted_poses

    def _filter_poses(self, poses, score_threshold=0.15):
        filtered_poses = []
        for pose in poses:
            if pose['score'] < score_threshold:
                continue

            filtered_poses.append(pose)
        
        return filtered_poses

    def predict(self, frame):
        # scale down the frame, normalize the pixels
        input_image, scale = self._process_frame(frame)

        # run the image through our network
        heatmaps_out, offsets_out, displacement_fwd_out, displacement_bwd_out = self._sess.run(
            [self._heatmaps, self._offsets,  self._displacement_fwd, self._displacement_bwd],
            feed_dict={'image:0': input_image}
        )

        # decode the output of the network 
        pose_scores, keypoint_scores, keypoint_coords = decode.decode_multiple_poses(
            heatmaps_out.squeeze(axis=0),
            offsets_out.squeeze(axis=0),
            displacement_fwd_out.squeeze(axis=0),
            displacement_bwd_out.squeeze(axis=0),
            output_stride=16,
            max_pose_detections=10,
            score_threshold=0.15,
            min_pose_score=0.15)

        # convert the decoded output to a nice json dictionary
        raw_poses = self._convert_raw_output_to_poses(pose_scores, keypoint_scores, keypoint_coords, scale)

        poses = self._filter_poses(raw_poses)

        return poses
