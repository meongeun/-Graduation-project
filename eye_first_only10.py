import cv2
import numpy as np
import dlib
from math import hypot
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import os
import numpy as np
import sys
from threading import Thread
import importlib.util
import serial 
port = "/dev/ttyACM0"
serialFromArduino = serial.Serial(port, 9600)
serialFromArduino.flushInput()
a=0

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret1 = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret1 = self.stream.set(3,resolution[0])
        ret1 = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
ap.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
ap.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
ap.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
arg1 = ap.parse_args()
args = vars(ap.parse_args())

MODEL_NAME = arg1.modeldir
GRAPH_NAME = arg1.graph
LABELMAP_NAME = arg1.labels
min_conf_threshold = float(arg1.threshold)
imW, imH = 640, 480
use_TPU = arg1.edgetpu

pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])
print("[INFO] starting video stream...")

#cap = VideoStream(src=0).start()
cap = cv2.VideoCapture("eye_recording.flv")
#cap = cv2.VideoCapture("eye2.mp4")
#cap2 = VideoStream(src=0).start()
#cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
vs = VideoStream(framerate=30).start()

time.sleep(2.0)
font = cv2.FONT_HERSHEY_PLAIN
while True:
    ret, frame = cap.read()
    frame2 = vs.read()
    if ret is False:
        break

    roi = frame[269: 795, 537: 1416]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 15, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if(len(contours) ==0):
        cv2.putText(roi, "BLINKING", (50, 150), font, 7, (255, 0, 0))
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if(M['m00']==0):
            cv2.putText(roi, "BLINKING", (50, 150), font, 7, (255, 0, 0))
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cx = round(2.64*cx - 34.372)
            #cy = round(2.644*cy)
            #cx = round(3.72*cx - 874.2)
            #cy = round(4.57*cy - 4.57)
            cv2.circle(roi, (cx, cy), 5, (0,0,255), -1)
            cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)
            if(cx>=imW/2 and cy>=imH/2):
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(roi, "LEFT TOP", (50, 150), font, 7, (255, 0, 0))
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)
            elif(cx>=imW/2 and cy<=imH/2):
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(roi, "LEFT BOTTOM", (50, 150), font, 7, (255, 0, 0))
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)
            elif(cx<=imW/2 and cy>=imH/2):
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)                   
                cv2.putText(roi, "RIGHT TOP", (50, 150), font, 7, (255, 0, 0)) 
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)     
            else:
                cv2.putText(frame2, "RIGHT BOTTOM", (50, 150), font, 7, (255, 0, 0))                    
                cv2.circle(frame2, (cx, cy), 5, (0,0,255), -1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            
        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    frame2 = vs.read()
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    boxes1 = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes1 = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores1 = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    for i in range(len(scores1)):
        if ((scores1[i] > min_conf_threshold) and (scores1[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes1[i][0] * imH)))
            xmin = int(max(1,(boxes1[i][1] * imW)))
            ymax = int(min(imH,(boxes1[i][2] * imH)))
            xmax = int(min(imW,(boxes1[i][3] * imW)))                                     
            cv2.rectangle(frame2, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            object_name = labels[int(classes1[i])] # Look up object name from "labels" array using class index
            
            if(xmin<= cx <=xmax and ymin <=cy<= ymax):
                if(object_name == 'clock'):
                    serialFromArduino.write('a'.encode())
                elif(object_name == 'cell phone'):
                    serialFromArduino.write('e'.encode())
                elif(object_name == 'book'):
                    serialFromArduino.write('b'.encode())
                elif(object_name == 'tv'):
                    serialFromArduino.write('c'.encode())
            label = '%s: %d%%' % (object_name, int(scores1[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame2, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame2, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    cv2.putText(frame2,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]

                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
        cv2.rectangle(frame2, (left, top), (right, bottom), (0, 255, 0), 2)
        if(name == 'JihoonKIm'):
            if(left<=cx<=right and bottom <= cy <=top):                    
                serialFromArduino.write('d'.encode())
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame2, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Frame2", frame2)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
vs.stop()