LICENSE_PLATE_DETECTOR_MODEL_PATH = 'model/plate_detector.onnx'
COCO_MODEL_PATH = 'model/yolo11s.pt'
VIDEOS_PATH = [ 'videos/street_1.mp4',
                'videos/street_2.mp4'
            ]
CLASS_OF_INTEREST = [1,2,5,7]

def image_crop(origin_image, bbox):

    x1, y1, x2, y2 = bbox

    croped_image = origin_image[int(y1):int(y2), int(x1):int(x2)]

    return croped_image

