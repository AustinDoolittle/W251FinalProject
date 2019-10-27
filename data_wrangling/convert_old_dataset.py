""" Converts the preexisting dataset into a joint directory 
Original Dataset location: https://www.amarchenkova.com/2018/12/04/data-set-convolutional-neural-network-yoga-pose/
"""

import os
from collections import defaultdict
import json

from PIL import Image

class_stats = defaultdict(int)
base_dir = './data/'
ignore_files = ['.DS_Store', '_DS_Store']
out_dir = os.path.join(base_dir, 'out')
os.makedirs(out_dir, exist_ok=True)
dirs = ['training_set', 'test_set']

for set_dir in dirs:
    base_path = os.path.join(base_dir, set_dir)
    for class_dir in os.listdir(base_path):
        in_path = os.path.join(base_path, class_dir)
        out_path = os.path.join(out_dir, class_dir)
        os.makedirs(out_path, exist_ok=True)
        for img in os.listdir(in_path):
            if img in ignore_files:
                continue

            img_number = class_stats[class_dir]
            origin_path = os.path.join(in_path, img)

            img = Image.open(origin_path)
            dest_path = os.path.join(out_dir, class_dir, str(img_number) + '.jpg')
            img.convert('RGB').save(dest_path, 'JPEG')
            class_stats[class_dir] += 1

with open(os.path.join(base_dir, 'class_counts.json'), 'w') as fp:
    json.dump(class_stats, fp)

