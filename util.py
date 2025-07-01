LICENSE_PLATE_DETECTOR_MODEL_PATH = 'model/plate_detector.onnx'
COCO_MODEL_PATH = 'model/yolo11m.pt'
VIDEOS_PATH = [ 'videos/street_1.mp4',
                'videos/street_2.mp4'
            ]
CLASS_OF_INTEREST = [1,2,5,7]

from ultralytics import YOLO

license_detector_model = YOLO(LICENSE_PLATE_DETECTOR_MODEL_PATH)

import easyocr

reader = easyocr.Reader(["th"])

def image_crop(origin_image, bbox):

    x1, y1, x2, y2 = bbox

    croped_image = origin_image[int(y1):int(y2), int(x1):int(x2)]

    return croped_image

def bbox_contains(bigger_box, smaller_box):
    bx1, by1, bx2, by2 = bigger_box
    sx1, sy1, sx2, sy2 = smaller_box
    return sx1 >= bx1 and sy1 >= by1 and sx2 <= bx2 and sy2 <= by2

def bbox_combine(boxA, boxB):
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    return [xA, yA, xB, yB]

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def find_adjacent_object(object_dict, current_object, classes=None, min_iou=0.1):
    for check_object in object_dict.values():
        if classes is not None:
            if check_object['cls'] not in classes:
                continue

        iou = bbox_iou(check_object['box'], current_object['box'])

        if iou > min_iou:
            return check_object
    
    return None

thai_consonants = [
    'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง',
    'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ',
    'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ',
    'ด', 'ต', 'ถ', 'ท', 'ธ', 'น',
    'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม',
    'ย', 'ร', 'ล', 'ว',
    'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ'
]

def plate_numbers_clean(number):

    number = number.replace(' ', '')

    return list(filter(lambda x: x.isdigit() or x in thai_consonants, number))

def get_plate_number_from_image(image):

    return reader.readtext(image)
