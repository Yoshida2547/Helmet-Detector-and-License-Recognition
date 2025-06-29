from ultralytics import YOLO
from ultralytics import solutions
from util import *
from keypoints import *

import numpy as np

import cv2 as cv
import cvzone

import threading

def handle_object_detection():

    cap = cv.VideoCapture(VIDEOS_PATH[0]) 

    # Check if the video file or camera was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    else:
        width  = cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`

        width   = int(width)
        height  = int(height)

    top_m,  bot_m   = (100, 100)
    left_m, rigth_m = (100, 100)

    region_points = [(left_m,top_m), (width-rigth_m, top_m), (width-rigth_m, height-bot_m), (left_m, height-bot_m)]

    tracker = solutions.TrackZone(
        region=region_points,
        model=COCO_MODEL_PATH
    )

    # Loop through frames
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If the frame was not read successfully (end of video), break the loop
        if not ret:
            print("End of video stream.")
            break

        results = tracker(frame)

        plot_frame = results.plot_im

        # Display the frame
        cv.imshow('Video Player', frame)
        cv.imshow('Ploted', plot_frame)

        # Wait for a key press (1 millisecond delay)
        # Press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and destroy all windows
    cap.release()
    cv.destroyAllWindows()

def main():
    object_detection_handler = threading.Thread(target=handle_object_detection)
    object_detection_handler.start()

    #while True:
    #    print(motorcycle_conf_dict)

if __name__ == '__main__':
    main()