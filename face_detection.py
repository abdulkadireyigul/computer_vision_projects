import face_recognition
import cv2
import numpy as np

data = {
    'aziz_sancar': './data/aziz_sancar.jpg',
    'oktay_sinanoglu': './data/oktay_sinanoglu.jpg',
    'feryal_ozel': './data/feryal_ozel.jpg'
}

known_face_encodings = []
known_face_names = []

for name, path in data.items():
    # NOTE: if the image includes multiple faces, face_encodings() will return an empty list
    # img = face_recognition.load_image_file('./data/group.jpg')
    # encodings = face_recognition.face_encodings(img)
    # print(encodings)

    img = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(img)[0]
    
    known_face_encodings.append(encoding)
    known_face_names.append(name)
    
# print(known_face_encodings)
# print(known_face_names)

# ========================= IMAGE =========================

target_frame = face_recognition.load_image_file('./data/feryal_ozel_unknown.jpg')

face_locations = face_recognition.face_locations(target_frame)
# print(face_locations, len(face_locations))
face_encodings = face_recognition.face_encodings(target_frame, face_locations)
# print(face_encodings, len(face_encodings))

# cv2.imshow('Original', target_frame)
target_frame = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR) # Convert the image to a OpenCV-compatible format
# cv2.imshow('Converted', target_frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# NOTE: zip function is used to iterate over two lists simultaneously
# example:
# names = ['John', 'Alice', 'Bob', 'Lucy']
# scores = [85, 90, 78, 92]
# res = zip(names, scores)
# print(list(res))
# [('John', 85), ('Alice', 90), ('Bob', 78), ('Lucy', 92)]

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # name = 'Unknown'
    # print(matches)
    
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        
        cv2.rectangle(target_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(target_frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(target_frame, name, (left + 2, bottom - 4), font, 0.4, (255, 255, 255), 1)
    
cv2.imshow('Image', target_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ========================= VIDEO =========================

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
    
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
            
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 2, bottom - 4), font, 0.4, (255, 255, 255), 1)
    
#     cv2.imshow('Video', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows

