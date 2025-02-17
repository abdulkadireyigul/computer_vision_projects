import os
import traceback

import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont

from anti_spoofing.test import test as anti_spoof_test

stack = traceback.extract_stack()
dirname = os.path.dirname(stack[-1].filename)

ANTI_SPOOF_MODEL_PATH = os.path.join(dirname, 'anti_spoofing', 'resources', 'anti_spoof_models')
print(f"ANTI_SPOOF_MODEL_PATH: {ANTI_SPOOF_MODEL_PATH}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device) # to detect faces

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # to extract face embeddings

data = {
    'aziz_sancar': './data/aziz_sancar.jpg',
    'oktay_sinanoglu': './data/oktay_sinanoglu.jpg',
    'feryal_ozel': './data/feryal_ozel.jpg',
    'me': './data/me.jpg',
}

known_face_encodings = []
known_face_names = []

for name, path in data.items():
    img = Image.open(path)
    boxes, _ = mtcnn.detect(img)
    
    if boxes is not None:
        aligned = mtcnn(img)
        embeddings = resnet(aligned).detach()
        known_face_encodings.append(embeddings)
        known_face_names.append(name)
        
def crop_image_with_ratio(img, height, width, middle):
    # source: https://github.com/KeerthiGowdaHN/Face-recognition-attendance-with-anti-spoofing/blob/main/app.py#L117
    h, w = img.shape[:2]
    h = h - h%4
    new_w = int(h / height) * width
    startx = middle - new_w // 2
    endx = middle + new_w // 2
    if startx <= 0:
        cropped_img = img[0:h, 0:new_w]
    elif endx >= w:
        cropped_img = img[0:h, w - new_w:w]
    else:
        cropped_img = img[0:h, startx:endx]
    return cropped_img

# target_frame = Image.open('./data/target.jpg')
# boxes, _ = mtcnn.detect(target_frame)

# if boxes is not None:
#     aligned = mtcnn(target_frame)
#     embeddings = resnet(aligned).detach()
    
#     # for box, embedding in zip(boxes, embeddings):
#     for i, (box, embedding) in enumerate(zip(boxes, embeddings)):
#         print(f"Face {i}: {box.tolist()} -> Embedding norm: {embedding.norm().item()}")
#         dists = [(e1 - embedding).norm().item() for e1 in known_face_encodings]
        
#         face = target_frame.crop(box.tolist())
#         face = crop_image_with_ratio(np.array(face), 4, 3, (box[0] + box[2])//2)

#         # face = np.array(face)
#         # face = Image.fromarray(face)
#         # face.show()
        
#         label, value = anti_spoof_test(np.array(face), ANTI_SPOOF_MODEL_PATH, device)
#         print(f"The face is {'real' if label <= 1 else 'spoof'} with {value:.2f} score.")
        
#         # draw_test = ImageDraw.Draw(face)
#         # draw_test.text((0, 0), f"{'real' if label <= 1 else 'spoof'}\n{value:.2f}", fill='black', font=ImageFont.truetype("arial.ttf", 20))
#         # face.show()
        
#         # compare face embeddings with known faces if the face is real
#         if min(dists) < 0.9 and label <= 1:
#             idx = dists.index(min(dists))
#             name = known_face_names[idx]
        
#             draw = ImageDraw.Draw(target_frame)
#             draw.rectangle(box.tolist(), outline='green', width=3)
#             draw.text((box[0], box[1]), name, fill='black', font=ImageFont.truetype("arial.ttf", 20))
            
#     target_frame.show()
    
# else:
#     print('No faces detected')

# ========================= VIDEO =========================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    
    boxes, _ = mtcnn.detect(frame)
    
    if boxes is not None:
        aligned = mtcnn(frame)
        embeddings = resnet(aligned).detach()
        
        for i, (box, embedding) in enumerate(zip(boxes, embeddings)):
            # print(f"Face {i}: {box.tolist()} -> Embedding norm: {embedding.norm().item()}")
            dists = [(e1 - embedding).norm().item() for e1 in known_face_encodings]
            
            face = frame.crop(box.tolist())
            face = crop_image_with_ratio(np.array(face), 4, 3, (box[0] + box[2])//2)
            
            # if ratio is not 4:3, then the image is not appropriate
            if face.shape[1]/face.shape[0] != 3/4:
                continue
            
            label, value = anti_spoof_test(np.array(face), ANTI_SPOOF_MODEL_PATH, device)
            # print(f"The face is {'real' if label <= 1 else 'spoof'} with {value:.2f} score.")
            
            # compare face embeddings with known faces if the face is real
            if min(dists) < 0.9 and label <= 1:
                idx = dists.index(min(dists))
                name = known_face_names[idx]
            
                draw = ImageDraw.Draw(frame)
                draw.rectangle(box.tolist(), outline='green', width=3)
                draw.text((box[0], box[1]), name, fill='black', font=ImageFont.truetype("arial.ttf", 20))
                
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()