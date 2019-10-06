import tensorflow as tf
import cv2

def main():
    print('Opening video capture')
    cap = cv2.VideoCapture(0)
    print('successfully opened')
    
    c = 0
    while True:
        res, frame = cap.read()
        c += 1

        if not res:
            logger.info('Failed to grab frame {c}')
            continue

        cv2.imshow("person!", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
    