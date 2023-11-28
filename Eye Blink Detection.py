import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from google.colab.patches import cv2_imshow
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap=cv2.VideoCapture("blinking.mp4")
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('output.mp4', fourcc, 29, (1080,1920))
font=cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def euclidean_distance(leftx,lefty, rightx, righty):
  return np.sqrt((leftx-rightx)**2 +(lefty-righty)**2)

def get_EAR(eye_points, facial_landmarks):
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y] 
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))  
    hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0), 3)
    ver_line = cv2.line(frame, (center_top[0], center_top[1]),(center_bottom[0], center_bottom[1]), (255, 0, 0), 3) 
    hor_line_lenght = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_lenght = euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])   
    EAR = ver_line_lenght / hor_line_lenght
    return EAR

eye_blink_signal=[]
blink_counter = 0
previous_ratio = 100
while True:
  ret, frame = cap.read() 
  if ret == False:
    break
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)
  for face in faces:
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    landmarks = predictor(gray, face)
    left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)
    right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
    blinking_ratio_1 = blinking_ratio * 100
    blinking_ratio_2 = np.round(blinking_ratio_1)
    blinking_ratio_rounded = blinking_ratio_2 / 100
    eye_blink_signal.append(blinking_ratio)
    if blinking_ratio < 0.20:
      if previous_ratio > 0.20:
        blink_counter = blink_counter + 1
    previous_ratio = blinking_ratio
  
  cv2.putText(frame, str(blink_counter), (30, 50), font, 2, (0, 0, 255),5)
  cv2.putText(frame, str(blinking_ratio_rounded), (900, 50), font, 2, (0, 0, 255),5)
  out.write(frame)
out.release()