# real-time-emotion-classifier

This is a real time emotion classifier using keras with tensorflow back-end and openCV. Opencv is used with haar cascade
to find faces in a frame. The face frame is then passed to the keras emotion classifier running in a seperate thread,
updating the predicted label. The label can also be exchanged for an emoji corresponding to the classified emotion.

A modified version of VGG16 is used with a different number of output classes. No existing dataset has been used for
training. Instead the emotion network is trained with own pictures by running get_training_data.py which starts a video
capture with the webcam and saves only the face frame.

## Instructions
Clone the repository to a targetDirectory

#### Install on Mac (Linux should be similar but not tested)

Originally made for Recommended installation in a python2.7 virtualenv with tensorflow, keras and openCV

##### Install tensorflow in a virtualenv\
1. sudo easy_install pip                 # if pip is not installed
2. pip install --upgrade virtualenv      
3. virtualenv --system-site-packages targetDirectory # for Python 2.7
4. cd targetDirectory
5. source ./bin/activate
6. easy_install -U pip
7. pip install --upgrade tensorflow      # for Python 2.7

##### Install requirements listed in REQUIREMENTS.txt
1. pip install requirements.txt

##### Or manually:
1. pip install keras
2. pip install opencv-python
3. pip install Pillow

#### Install on windows
Installing on windows should be similar to installing on mac except you will have to use python 3.5
##### Install tensorflow in a virtualenv\
1. sudo easy_install pip                 # if pip is not installed
2. pip3 install --upgrade virtualenv 
3. virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
4. cd targetDirectory
5. source ./bin/activate
6. easy_install -U pip
7. pip3 install --upgrade tensorflow     # for Python 3.n

##### Install requirements listed in REQUIREMENTS.txt
1. pip3 install requirements.txt

### Creating training data

By running get_training_data.py the default webcam will start to capture face frames and displaying a preview of the
captured frame. The image_count variable decides how many images that will be taken for each label. The label_name will
only appear in the file name to indicate the label.


1. Decide how many facial expressions you want the classifier to recognize. Create a 
new directory called 'dataset' at project root and then create three directories called 'train_data', 'valid_data' and 
 'test_data'. All three directories should contain one directory for each label name. 
  For example with label names 'neutral' and 'smile' the directory structure would look like this:\
dataset/\
├── test_data\
│   ├── neutral\
│   └── smile\
├── train_data\
│   ├── neutral\
│   └── smile\
└── valid_data\
    ├── neutral\
    └── smile

2. Run get_training_data.py with the name_label set to neutral and sit in front of your webcam with a neutral look.
For better results try moving your head to get more angles and moving around to capture different lighting conditions.
3. The captured images are saved in the dataset directory. Look through the images to make sure that all images captured 
your face correctly. 
4. (If some images was captured incorrectly) select every image that was captured incorrectly (ctr/cmd click) and copy
the files and paste the filenames into /dataset/deleted_images.txt. Then run retake_lost_data.py. Repeat this step until 
all images contains a face. 
5. Divide the images into test, train, and validation data. A typical split can be 60% training data 20% test and 20% 
validation data. For example if 200 images where taken, select 120 images and drag them into train_data directory. Drag 
40 images into test_data and 40 images into valid_data. 
6. Place the images into the directory with the correct label. 
7. Repeat steps 2-6 for every facial expression in the model.   

### Training the model

1. If you are following the example with 200 images for every label (neutral and smile) you can simply run 
train_emotion_model.py. 
2. If training the model with different parameters, change the batch_size, validation_steps, steps_per_epoch and epochs
accordingly. 
3. To train with more classes you will need to make sure the last dense layer has the correct output and that 
class_labels contains all the necessary labels.

### Running the real time classifier
1. Open real-time-webcam-demo.py.
2. To run only the classifier, make sure the emoji_mode variable is set to false.
3. Run the script. 
