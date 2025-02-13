import cv2
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# util function to detect text from image
def detect_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(Image.fromarray(thresholded), config='--psm 6')
    text = text.strip()
    
    return text

# ======================== Main ========================

# vid = cv2.VideoCapture(0)

# while True:
#     ret, frame = vid.read()
    
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
    
#     text = detect_text(frame)
    
#     if len(text) > 0:
#         print(text)
    
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(24) & 0xFF == ord('q'):
#         break
    
# vid.release()
# cv2.destroyAllWindows()

# ======================== Test ========================

img = cv2.imread('./data/text.jpg')

if img is None:
    print("Can't open image file")
    exit(0)
    
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('thresholded', thresholded)

# thresholded = cv2.adaptiveThreshold(
#     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )
# cv2.imshow('thresholded 2', thresholded)

# thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('thresholded 3', thresholded)

# Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(gray, (5,5), 0)
# _, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('thresholded 4', thresholded)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

text = detect_text(img)
if len(text) > 0:
    print(text)