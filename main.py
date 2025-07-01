from ultralytics import YOLO
from ultralytics import solutions
from util import *

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

    top_m,  bot_m   = (0, 0)
    left_m, rigth_m = (0, 0)

    region_points = [(left_m,top_m), (width-rigth_m, top_m), (width-rigth_m, height-bot_m), (left_m, height-bot_m)]

    tracker = solutions.TrackZone(
        region=region_points,
        model=COCO_MODEL_PATH,
        classes=[3, 0],
        tracker='botsort.yaml',
        conf=0.25,
        verbose=False
    )

    past_object_dict = None

    capture_y = 300

    print(tracker.names)

    # Loop through frames
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If the frame was not read successfully (end of video), break the loop
        if not ret:
            print("End of video stream.")
            break

        results = tracker.process(frame.copy())

        plot_frame = results.plot_im # type: ignore

        cv.line(plot_frame, (0, capture_y), (width, capture_y), color=(0,255,0), thickness=2) # type: ignore

        post_object_dict = {
            id:{
                'id':id,
                'box':box.numpy(),
                'cls':tracker.names[int(cls)],
                'pos':box.numpy()[:2],
                'conf':conf
            }

            for id, box, cls, conf in zip(tracker.track_ids, tracker.boxes, tracker.clss, tracker.confs)
        }

        for post_object in post_object_dict.values():

            if post_object['cls'] != 'motorcycle':
                continue

            if post_object['id'] not in past_object_dict or past_object_dict is None:
                continue

            past_object = past_object_dict[post_object['id']]

            post_pos = post_object['pos']
            past_pos = past_object['pos']

            if post_pos[1] < capture_y and past_pos[1] > capture_y:

                bbox = post_object['box']

                person = find_adjacent_object(post_object_dict, post_object, classes=['person'])

                if person is not None:
                    bbox = bbox_combine(bbox, person['box'])

                image = image_crop(frame, bbox)

                cv.imshow(f'{post_object['id']}', image)
                print(post_object)

                pass

        past_object_dict = post_object_dict.copy()

        # Display the frame
        cv.imshow('Result', plot_frame) # type: ignore

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