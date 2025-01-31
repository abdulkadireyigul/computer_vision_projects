import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# img = cv2.imread('person.jpg', cv2.IMREAD_GRAYSCALE) # OR 0

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# =================================================================================================

# img = cv2.imread('person.jpg', 0)

# cv2.imwrite('person_gray.jpg', img)

# =================================================================================================

# vid = cv2.VideoCapture(0)

# while True:
#     ret, frame = vid.read()
#     cv2.imshow('frame', frame)
    
#     # kInp = cv2.waitKey(1)
    
#     # if kInp == ord('q'):
#     #     break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# vid.release()
# cv2.destroyAllWindows()

# =================================================================================================

# cap = cv2.VideoCapture('video.mp4')

# if (cap.isOpened() == False):
#     print('Error opening video file')

# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
    
#     if ret == False:
#         break

#     if cv2.waitKey(24) & 0xFF == ord('q'): # 24 is the frame rate
#         break
    
# cap.release()
# cv2.destroyAllWindows()

# =================================================================================================

# vid = cv2.VideoCapture(0)

# w = int(vid.get(3))
# h = int(vid.get(4))

# size = (w, h)

# result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, size)

# while True:
#     ret, frame = vid.read()
    
#     if ret == False:
#         break

#     cv2.imshow('frame', frame)
#     result.write(frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         break
    
# vid.release()
# result.release()
# cv2.destroyAllWindows()

# print('Video saved successfully')

# =================================================================================================