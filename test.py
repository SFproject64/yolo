from ctypes import *
import os                                               # Import libraries
import cv2
import numpy as np
import PIL
import io
import time
import matplotlib.pyplot as plt
import darknet


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'mask' : [0, 255, 0], 'nomask': [255, 0, 0], 'object' : [255, 255, 0]
    }
    label_dict = { 
        'mask': 'Wearing Mask', 'nomask' : 'Don\'t Wearing Mask', 'object' :'Object covering the face'
    }
    
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        name_tag = detection[0]
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val 
                label = label_dict[name_key]
                xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                # round_str = round(float(detection[1]) * 100, 2)
                cv2.putText(img,
                            label + "  [ " + detection[1] + " % ] ",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)

  #  print()       

    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
   
    global metaMain, netMain, altNames, class_names
    configPath = "./cfg/yolov4-custom1.cfg"                                 # Path to cfg
    weightPath = "./training/yolov4-custom_best.weights"                                 # Path to weights
    metaPath = "./data/obj.data"                                    # Path to meta data
    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode( 
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
        class_names = [metaMain.names[i].decode("ascii") for i in range(metaMain.classes)]
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(1)                                      # Uncomment to use Webcam
    #cap = cv2.VideoCapture("test2.mp4") 
                                # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video writer
    out = cv2.VideoWriter(                                          # Set the Output path for video writer
        "./Demo/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (frame_width, frame_height))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, class_names, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Faced Mask Detection', image)                                    # Display Image window
        if cv2.waitKey(1)==27:                                  # Press esc to stop webcam
            break
        out.write(image)                                             # Write that frame into output video
    cap.release()                                                    # For releasing cap and out. 
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":  
    YOLO() 
