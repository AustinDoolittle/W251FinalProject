import os
import glob
import json
from collections import defaultdict

import numpy as np
import cv2

BASE_DIR = './data/out'

def image_equals(lhs, rhs, threshold=1e-3):
    if lhs.shape != rhs.shape:
        return False
    subt_np = np.subtract(lhs,rhs)
    mean_np = np.mean(subt_np)

    return mean_np < threshold

class_dirs = list(os.listdir(BASE_DIR))
class_images = defaultdict(list)
for class_dir in class_dirs:
    class_path = os.path.join(BASE_DIR, class_dir)

    # remove all PNGs, they're dupes from an early iteration of the scraper
    png_count = 0
    for png_img in glob.glob(os.path.join(class_path, '*.png')):
        print(f'Removing {png_img} because PNG')
        os.remove(png_img)

    img_files = list(os.listdir(class_path))
    for i, img_file in enumerate(img_files):
        image_path = os.path.join(class_path, img_file)
        new_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if new_image is None:
            print(f'Removing {image_path} because failed to load')
            os.remove(image_path)
            continue

        is_dupe = False
        for rhs in class_images[class_dir]:
            other_image = rhs['image']
            if image_equals(new_image, other_image):
                # cv2.imshow('new_image', new_image)
                # cv2.waitKey(0)
                # cv2.imshow('other_image', rhs)
                # cv2.waitKey(0)
                is_dupe = True
                print(f"Deleting {image_path} because dupe of {rhs['filepath']}")
                break

        if is_dupe:
            os.remove(image_path)
            pass
        else:
            class_images[class_dir].append({
                'image': new_image,
                'filepath': image_path,
            })
    
# do a final pass over all images to identify duplicates across classes
class_keys = list(class_images.keys())
for i, curr_key in enumerate(class_keys):
    curr_images = class_images[curr_key]
    for other_key in class_keys[i:]:
        if curr_key == other_key:
            continue

        other_images = class_images[other_key]
        for curr_img in curr_images:
            for other_img in other_images:
                if image_equals(curr_img['image'], other_img['image']):
                    print(f"DUPE lhs class: {curr_key}, lhs filepath: {curr_img['filepath']}; rhs class: {other_key}, rhs filepath: {other_img['filepath']}")
                    os.remove(curr_img['filepath'])
                    os.remove(other_img['filepath'])
