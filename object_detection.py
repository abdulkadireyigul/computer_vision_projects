import cv2
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv3u model
model = YOLO("yolov3u.pt")

def detect_object(frame):
    """
    Detect objects in a frame using YOLOv3u model.
    
    Args:
        frame: A frame from a video stream or an image.
        
    Returns:
        A list of detected objects with their bounding box coordinates, class, and confidence level.
    """
    results = model(frame)
    
    boxes, cls, conf = results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
    names = results[0].names
    
    result = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        obj = names[int(cls[i])]
        conf_level = conf[i] * 100
        
        result.append([x1, y1, x2, y2, obj, conf_level])
        
    return result

# =================== Main Image ===================

img = cv2.imread("./data/group.jpg")

results = detect_object(img)

for result in results:
    x1, y1, x2, y2, obj, conf_level = result
    
    title = obj + " " + str(int(conf_level)) + "%"
    
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.putText(img, title, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# =================== Main Video Stream ===================

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     results = detect_object(frame)
    
#     for result in results:
#         x1, y1, x2, y2, obj, conf_level = result

#         title = obj + " " + str(int(conf_level)) + "%"

#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#         cv2.putText(frame, title, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     cv2.imshow("frame", frame)
    
#     if cv2.waitKey(24) & 0xFF == ord("q"):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

