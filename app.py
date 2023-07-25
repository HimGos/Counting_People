## Importing Libraries
from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import numpy as np


# Model declaration
model = YOLO('yolov8l.pt')

# Creating video capture object
cap = cv2.VideoCapture('escalator.mp4')

# Classes
classnames = ["person", "bicycle", 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
              'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissor', 'teddy bear', 'hair drier', 'toothbrush']

# Masking, used for masking video to focus on specific region
mask = cv2.imread('mask.png')

# Initializing the object tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Coordinates of first floor line used for counting object
first_floor_coord = [1425, 900, 1600, 900]

# Coordinates of second floor line used for counting object
second_floor_coord = [400, 650, 550, 650]

# List to store the unique IDs of objects that have passed through the first floor counting line
first_floor_count = []

# List to store the unique IDs of objects that have passed through the second floor counting line
second_floor_count = []

while True:
    success, img = cap.read()
    if not success:
        break

    # Focus on specific regions of the frame using mask
    img_region = cv2.bitwise_and(src1=img, src2=mask)

    results = model(img_region, stream=True)

    # Top left graphic image counter
    img_graphic = cv2.imread('graphic.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(imgBack=img, imgFront=img_graphic, pos=(0, 0))

    # Store the bounding box coordinates and ID of detected object.
    detections = np.empty((0, 5))

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            # Confidence score
            conf = round(float(box.conf[0]), 2)

            # Class object belongs to
            cls = classnames[int(box.cls[0])]

            if cls=='person':
                # cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), l=10)
                # cvzone.putTextRect(img=img, text=f'{cls} | {conf}',
                #                 pos=(max(0, x1), max(35, y1)),
                #                 scale=1,
                #                 thickness=1,
                #                 offset=5,
                #                 colorT=(0, 0, 0),
                #                 colorR=(0, 255, 255),
                #                 colorB=(255, 255, 0))
                    
                # Bounding box and confidence is added to detections array
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    ## OBJECT TRACKING
    # Detected object info is passed to tracker
    result_tracker = tracker.update(detections)

    # Creating first floor counting line
    cv2.line(img=img,
             pt1=(first_floor_coord[0], first_floor_coord[1]),
             pt2=(first_floor_coord[2], first_floor_coord[3]),
             color=(0, 0, 225),
             thickness=5)
    
    # Creating second floor counting line
    cv2.line(img=img,
             pt1=(second_floor_coord[0], second_floor_coord[1]),
             pt2=(second_floor_coord[2], second_floor_coord[3]),
             color=(0, 0, 225),
             thickness=5)

    # Tracking the detected object
    for result in result_tracker:

        # For each tracked object, extracts the bounding box corrdinates and ID.
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        # Creating box only on tracked objects
        cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img=img,
                                text=f'{int(Id)}',
                                pos=(max(0, x1), max(35, y1)),
                                scale=2,
                                thickness=3,
                                offset=10,
                                colorT=(0, 0, 0),
                                colorR=(0, 255, 255),
                                colorB=(255, 255, 0))
        
        # Creating centroid within box
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img=img, center=(cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        # checks if the centroid of the object crosses the first floor  counting line OR crosses the second floor counting line
        if (first_floor_coord[0]< cx < first_floor_coord[2] and first_floor_coord[1]-35 < cy < first_floor_coord[1]+35):

            # checks if the onject ID is not already in the first floor list
            if first_floor_count.count(Id) == 0:
                # object just crossed the line for the first time, so the object ID is added to first_floor_counts.
                first_floor_count.append(Id)
                # Line color also changes when we touch it
                cv2.line(img=img, 
                        pt1=(first_floor_coord[0], first_floor_coord[1]), 
                        pt2=(first_floor_coord[2], first_floor_coord[3]), 
                        color=(0, 255, 255), 
                        thickness=5)
        

        elif (second_floor_coord[0]< cx < second_floor_coord[2] and second_floor_coord[1]-35 < cy < second_floor_coord[1]+35):
            
            # checks if the onject ID is not already in the second floor list
            if second_floor_count.count(Id) == 0: 
                # object just crossed the line for the first time, so the object ID is added to total_counts.
                second_floor_count.append(Id)
                # Line color also changes when we touch it
                cv2.line(img=img, 
                        pt1=(second_floor_coord[0], second_floor_coord[1]), 
                        pt2=(second_floor_coord[2], second_floor_coord[3]), 
                        color=(0, 255, 255), 
                        thickness=5)
                    
    # Showing fancy count at top left corner
    cv2.putText(img=img, text=str(len(first_floor_count)), org=(220, 45), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(136, 0, 216), thickness=3)
    
    # Showing fancy count at top left corner
    cv2.putText(img=img, text=str(len(second_floor_count)), org=(220, 90), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(136, 0, 216), thickness=3)

    # Final image
    cv2.imshow('Image', img)
    cv2.waitKey(1)

