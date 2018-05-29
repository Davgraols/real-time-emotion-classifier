import threading
import time
import sys
import cv2
import numpy as np
from keras.models import load_model

# for other models which needs custom compunents
from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
from keras.utils.generic_utils import CustomObjectScope


class PredictionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        global face_frame
        global running
        # Load the VGG16 network
        self.model = load_model('./../models/detect_emo_model.h5')
        """with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                                'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                            self.model = load_model('./../models/mobileNet_detect_emo_model.h5')"""
        print("[INFO] loading network...")

        while running:
            if not emoji_mode:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_frame = cv2.resize(frame[y:y + h, x:x + w], (224, 224))

                label = self.predict(cv2.resize(face_frame, (224,224)))
                time.sleep(1)
            else:
                label = self.predict(cv2.resize(face_frame, (224, 224)))
                time.sleep(1)

    def predict(self, predict_frame):
        image = cv2.cvtColor(predict_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image.reshape((1,) + image.shape)

        prediction = self.model.predict(image)

        return_label = np.argmax(prediction) # returns max index

        print(prediction, ": ", label_list[return_label])
        return return_label


cascPath = "./../models/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

smiling_emoji = cv2.imread("./../images/Smiling_Emoji.png")
#smiling_emoji = cv2.resize(smiling_emoji, (100, 100))
neutral_emoji = cv2.imread("./../images/neutral_emoji.png")
#neutral_emoji = cv2.resize(neutral_emoji, (100, 100))

emoji_list = [neutral_emoji, smiling_emoji]
label_list = ["neutral", "smile"]


emoji_mode = True

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
keras_thread = PredictionThread()
running = True
keras_thread.start()
label = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if emoji_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_frame = cv2.resize(frame[y:y + h, x:x + w], (224,224))

        emoji = emoji_list[label]
        emoji = cv2.resize(emoji, (w, h))
        frame[y:y + emoji.shape[0], x:x + emoji.shape[1]] = emoji

    # Display the resulting frame
    cv2.putText(frame, "Label: {}".format(label_list[label]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
running = False
video_capture.release()
cv2.destroyAllWindows()
sys.exit()
