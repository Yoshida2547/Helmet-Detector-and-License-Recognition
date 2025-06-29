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
        model=COCO_MODEL_PATH,
        classes=[3],
        tracker='botsort.yaml',
        conf=0.1,
        verbose=False
    )

    cross = 255

    capture_line_point = [(0,cross), (width,cross)]

    previous_boxes_dict = None

    # Loop through frames
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If the frame was not read successfully (end of video), break the loop
        if not ret:
            print("End of video stream.")
            break

        results = tracker.process(frame)

        plot_frame = results.plot_im # type: ignore

        cv.line(plot_frame, capture_line_point[0], capture_line_point[1], color=(0, 255, 0), thickness=2) # type: ignore

        # Display the frame
        cv.imshow('Video Player', frame)

        current_boxes_dict = { id:box.numpy() for id, box in zip(tracker.track_ids, tracker.boxes)}

        for id in list(current_boxes_dict.keys()):
            if id not in previous_boxes_dict or previous_boxes_dict is None:
                continue

            current_box     = current_boxes_dict[id]
            previous_box    = previous_boxes_dict[id]

            x1, y1 = previous_box[:2]
            x2, y2 = current_box[:2]

            if y1 > cross and y2 < cross:
                pass

        # Wait for a key press (1 millisecond delay)
        # Press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        previous_boxes_dict = current_boxes_dict.copy()

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