import os

import cv2

from model import PoseNetModel
from constants import *

def file_generator(input_dir):
    class_dirs = list(os.listdir(input_dir))

    for class_dir in class_dirs:
        class_dir_path = os.path.join(input_dir, class_dir)

        class_files = list(os.listdir(class_dir_path))
        for class_file in class_files:
            class_file_path = os.path.join(class_dir_path, class_file)
        
            yield (class_dir, class_file_path)


def main():
    model = PoseNetModel('/app/model.pb')

    for class_name, file_path in file_generator('/app/data/images/'):
        image = cv2.imread(file_path)
        poses = model.predict(image)
        print(len(poses))



if __name__ == '__main__':
    main()