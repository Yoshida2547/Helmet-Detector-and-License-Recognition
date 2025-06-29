from ultralytics import YOLO
from ultralytics import solutions
from util import *
from keypoints import *

import numpy as np

import cv2 as cv
import cvzone

import threading

coco_model = YOLO(COCO_MODEL_PATH)
plate_model = YOLO(LICENSE_PLATE_DETECTOR_MODEL_PATH)

motorcycle_dict = {}
motorcycle_conf_dict = {}

def handle_object_detection():

    for result in coco_model.track(source=VIDEOS_PATH[0], 
                                   stream=True, 
                                   conf=0.20, 
                                   iou=0.1,
                                   classes=[3, 2, 1, 0],
                                   persist=True,
                                   verbose=False,
                                   ):

        origin_image = result.orig_img
        ploted_image = result.plot()

        cv.imshow('original frame', ploted_image)

        boxes_motorcyle = [ box for box in result.boxes if result.names[int(box.cls)] == 'motorcycle'] # type: ignore

        for box in boxes_motorcyle:

            if box.id is None:
                break

            #crop the motorcycle image
            motor_bbox = box.xyxy.cpu().numpy()[0] # type: ignore
            motor_id = int(box.id.cpu().numpy())
            motor_conf = float(box.conf)

            crop_image = image_crop(origin_image, motor_bbox)

            print(motor_id)

            if motorcycle_dict.get(motor_id) is None:
                motorcycle_dict[motor_id] = Motorcycle(motor_id, crop_image, box)
                motorcycle_conf_dict[motor_id] = motor_conf

            elif motorcycle_conf_dict[motor_id] < motor_conf:
                motorcycle_dict[motor_id] = Motorcycle(motor_id, crop_image, box)
                motorcycle_conf_dict[motor_id] = motor_conf

        if cv.waitKey(25) & 0xFF ==  ord('q'):
            break
    
    cv.destroyAllWindows()

    return

def main():
    object_detection_handler = threading.Thread(target=handle_object_detection)
    object_detection_handler.start()

    #while True:
    #    print(motorcycle_conf_dict)

if __name__ == '__main__':
    main()