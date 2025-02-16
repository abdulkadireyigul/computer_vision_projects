import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device) # to detect faces
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # to extract face embeddings

data = {
    'aziz_sancar': './data/aziz_sancar.jpg',
    'oktay_sinanoglu': './data/oktay_sinanoglu.jpg',
    'feryal_ozel': './data/feryal_ozel.jpg',
    # 'me': './data/me.jpeg'
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

# target_frame = Image.open('./data/feryal_ozel_unknown.jpg')
# boxes, _ = mtcnn.detect(target_frame)

# if boxes is not None:
#     aligned = mtcnn(target_frame)
#     embeddings = resnet(aligned).detach()
    
#     for box, embedding in zip(boxes, embeddings):
#         dists = [(e1 - embedding).norm().item() for e1 in known_face_encodings]
        
#         # compare face embeddings with known faces
#         if min(dists) < 0.9:
#             idx = dists.index(min(dists))
#             name = known_face_names[idx]
            
#             draw = ImageDraw.Draw(target_frame)
#             draw.rectangle(box.tolist(), outline='red', width=3)
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
        
        for box, embedding in zip(boxes, embeddings):
            dists = [(e1 - embedding).norm().item() for e1 in known_face_encodings]
            
            if min(dists) < 0.9:
                idx = dists.index(min(dists))
                name = known_face_names[idx]
                
                draw = ImageDraw.Draw(frame)
                draw.rectangle(box.tolist(), outline='red', width=3)
                draw.text((box[0], box[1]), name, fill='black', font=ImageFont.truetype("arial.ttf", 20))
                
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# NOTE: will be updated soon for face anti-spoofing