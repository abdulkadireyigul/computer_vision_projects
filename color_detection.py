import cv2
import numpy as np
from PIL import Image

# Function to get the lower and upper limits of the color
# source: https://github.com/computervisioneng/color-detection-opencv/blob/master/util.py
def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

# =============== Main code ===============

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('./data/video.mp4')

# Define the color to detect
yellow = [0, 255, 255] # BGR values

target_color = yellow

while True:
    ret, frame = cap.read()
    
    # convert BGR to HSV (Hue, Saturation, Value), because it is easy to represent a color in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the lower and upper limits of the color
    lowerLimit, upperLimit = get_limits(target_color)
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, lowerLimit, upperLimit)
    
    mask_ = Image.fromarray(mask)
    
    bbox = mask_.getbbox()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()