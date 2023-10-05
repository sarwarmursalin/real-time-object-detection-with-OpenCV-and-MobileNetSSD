# real_time_object_detection.py
# To compile - $python3 real_time_object_detection.py -p pathtoprototxtfile -m pathtocaffemodelfile

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"] #list of class labels the nodel is trained to detect
IGNORE = set(["horse","train","cow","boat"]) #list of class labels we want to not detect
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) #generate bounding box colors



print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"]) #load model from disk


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start() #initialize video stream
#vs = cv2.VideoCapture('/home/adamyaa14/Desktop/real-time-object-detection/PrideAndPrejudice-JabWeMet.mp4')
time.sleep(2.0) #camera sensor warmup
fps = FPS().start() #initialise fps counter

# loop over the frames from the video stream
while True:
	frame = vs.read() #get the frame from the threaded video
	frame = imutils.resize(frame, width=800) #resize frame to 800 pixels

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5) #frame to blob

	
	net.setInput(blob) #pass the blob to the network
	detections = net.forward() #returns detection or the predictions

	
	for i in np.arange(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2] #probability of predictions

		
		if confidence > args["confidence"]: #filter weak detections
			
			idx = int(detections[0, 0, i, 1]) #index of the class label
			
			if CLASSES[idx] in IGNORE:
				continue
				
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("REAL TIME OBJECT DETECTION FRAME", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
