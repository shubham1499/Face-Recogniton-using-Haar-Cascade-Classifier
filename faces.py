import os
import cv2
import numpy as num
import pickle
######################################################
filename = 'video-gray.mp4'
frames_per_second = 24.0
my_res = '720p'

def change_res(cap,width, height):
    cap.set(3, width)
    cap.set(4, height)

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
   # 'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


cap = cv2.VideoCapture(0)
dims = get_dims(cap,res = my_res)
video_type_cv2 = get_video_type(filename)
out = cv2.VideoWriter(filename,video_type_cv2,frames_per_second,dims)

################################################################

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name":1}
with open("labels.pickle",'rb')as f:
	labels = pickle.load(f)
	labels = {v:k for k,v in labels.items()}


cap = cv2.VideoCapture(0)

while(True):
	#capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray ,scaleFactor=1.3, minNeighbors=5)
	for(x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w]
		roi_color= frame[y:y+h,x:x+w]
		id_, conf = recognizer.predict(roi_gray)

		if conf>=45 and conf<=85:
		 print(id_)
		 print(labels[id_])
		 font = cv2.FONT_HERSHEY_SIMPLEX
		 name = labels[id_]
		 color = (255,255,255)
		 stroke = 2
		 cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		 
		img_item = 'my_img.png'
		cv2.imwrite(img_item,roi_gray)
		color = (0, 0, 255)
		stroke = 2
		width = x + w
		height= y + h
		cv2.rectangle(frame, (x, y), (width, height), color, stroke)

	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

out.release()
cap.release()
cv2.destroyAllWindows()