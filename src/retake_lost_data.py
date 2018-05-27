import cv2


def find_face(full_frame):
    only_face_frame = full_frame
    gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        only_face_frame = frame[y:y + h, x:x + w]

    return only_face_frame

def load_lost_files():
    try:
        file = open("./../dataset/replace_images.txt", "r")
    except IOError:
        print 'cannot read file'

    file_names = []
    for line in file:
        file_names.append(line.replace("\n", ""))

    return file_names



cascPath = "./../models/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Start a video capture with default camera
video_capture = cv2.VideoCapture(0)


files = load_lost_files()

for file in files:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    face_frame = find_face(frame)
    # shows a preview of photo taken
    cv2.imshow('Video', face_frame)
    filename = "./../dataset/" + file

    try:
        face_frame = cv2.resize(face_frame, (224, 224))
        cv2.imwrite(filename, face_frame)
        print "file wrote: " + filename
    except IOError:
        print 'cannot write image'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
