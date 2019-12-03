import argparse
import logging
import sys
import time
import os

import tensorflow as tf
import numpy as np
import cv2

from constants import *
import decode
from model import PoseNetModel

import paho.mqtt.client as mqtt
import uuid
import json
import base64

MQTT_BROKER_OUT = 'pose_broker_1'
MQTT_TOPIC_OUT = 'edge_capture'
FANCY_PARTS = ["nose", "leftEye", "rightEye", "leftEar", "rightEar"]


def file_generator(input_dir):
    class_dirs = list(os.listdir(input_dir))

    for class_dir in class_dirs:
        class_dir_path = os.path.join(input_dir, class_dir)

        class_files = list(os.listdir(class_dir_path))
        for class_file in class_files:
            class_file_path = os.path.join(class_dir_path, class_file)
        
            yield (class_dir, class_file_path)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', required=True)

    return parser.parse_args(argv)


class FrameCounter:
    def __init__(self):
        self._count = 0
        self._prev_frame = 0
        self.fps = 0

    def tick(self):
        if not self._prev_frame:
            self._prev_frame = time.time()
            return

        now = time.time()
        spf = now - self._prev_frame
        self.fps = 1 / spf
        self._prev_frame = now

    def overlay_fps(self, img):
        if not self.fps:
            fps_text = '--'
        else:
            fps_text = '%.1f'%self.fps

        return cv2.putText(img, fps_text + ' FPS', (25, 25), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.5, (0, 255, 0), 2, cv2.LINE_AA) 


def get_keypoints(parts, threshold=0.5, fancy=False):
    keypoints = []
    for part in parts:
        if fancy and part in FANCY_PARTS:
            continue

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
    text = 'Score: %.3f'%pose_score
    return cv2.putText(frame, text, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

def overlay_poses(poses, frame, instance_score_threshold=0.5, part_score_threshold=0.5, fancy=False):
    keypoints = []
    lines = []
    for pose in poses:
        if pose['score'] < part_score_threshold:
            continue
        
        parts = pose['parts']
        if fancy:
            overlay_score(pose, frame)

        new_keypoints = get_keypoints(parts, threshold=part_score_threshold, fancy=fancy)
        keypoints += new_keypoints

        new_lines = get_lines(parts, threshold=part_score_threshold)
        lines += new_lines

    kp_frame = cv2.drawKeypoints(frame, keypoints, outImage=np.array([]))
    return cv2.polylines(kp_frame, lines, isClosed=False, color=(255,255,0))

def publish_poses(client_uuid, pose, frame):
    # TODO Wire up MQTT
    print('Pose found!') 
    pose_payload = {'client': client_uuid,
            'pose': pose,
            'image': frame}
    client = mqtt.Client(client_uuid)
    client.connect(MQTT_BROKER_OUT, 1883, 60)
    client.publish(MQTT_TOPIC_OUT, payload = json.dumps(pose_payload), qos = 1)
    client.disconnect()

def main(args):
    # Create our video capture device
    client_uuid = str(uuid.uuid1())
    print('Client UUID: {}'.format(client_uuid))
    print('Opening video capture')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    print('successfully opened')
    
    frame_counter = FrameCounter()
    model = PoseNetModel(args.model_file)

    # enter loop
    c = 0
    while True:
        # read a frame
        res, frame = cap.read()

        c += 1

        # check if we were successful
        if not res:
            print('Failed to grab frame %i'%c)
            continue
    
        frame_counter.tick()

        # scale down the frame, normalize the pixels
        poses = model.predict(frame)

        display_frame = frame.copy()
        display_frame = overlay_poses(
            display_frame, 
            frame, 
            instance_score_threshold=0.15, 
            part_score_threshold=0.15,
            fancy=True
        )
        display_frame = frame_counter.overlay_fps(display_frame)

        # show the people what we did
        cv2.imshow("person!", display_frame)
        cv2.waitKey(1)
                
        inference_frame = overlay_poses(
            poses, 
            frame,
            instance_score_threshold=0.15, 
            part_score_threshold=0.15,
            fancy=False
        )
        for pose in poses:
            if pose['score'] > .25 and pose['parts']['leftHip']['score'] > .25 and pose['parts']['rightHip']['score'] > .25:
                # sent our poses to the MQTT topic
                gray_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY)
                queue_frame = base64.b64encode(cv2.imencode('.jpg', gray_frame)[1]).decode()
                publish_poses(client_uuid, pose, queue_frame)



if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
    
