# real-time-emotion-classifier

This is a real time emotion classifier using keras with tensorflow back-end and openCV. Opencv is used with haar cascade
to find faces in a frame. The face frame is then passed to the keras emotion classifier running in a seperate thread,
updating the predicted label. The label can also be exchanged for an emoji corresponding to the classified emotion.

A modified version of VGG16 is used with a different number of output classes. No existing dataset has been used for
training. Instead the emotion network is trained with own pictures by running get_training_data.py which starts a video
capture with the webcam and saves only the face frame.

## Instructions

Recomended installation in a python2.7 virtualenv with tensorflow, keras and openCV

Install requirements listed in REQUIREMENTS.txt
pip install requirements.txt

Or manually:
pip install --upgrade tensorflow

pip install keras

pip install opencv-python

pip install Pillow

### Creating training data

By running get_training_data.py the default webcam will start to capture face frames and displaying a preview of the
captured frame. The image_count variable decides how many images that will be taken for each label. The label_name will
only appear in the file name to indicate the label.


1. Decide how many facial expressions you want the classifier to recognize. Create a 
new directory with the label name so that every directory in dataset contains a directory for each label. For example 
with label names 'neutral' and 'smile' the directory structure would look like this:\
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
3. When training is done and you have successfully saved the model to disk you can run the  


### Running the real time classifier
1. Open real-time-webcam-demo.py.
2. To run only the classifier, make sure the emoji_mode variable is set to false.
3. Run the script. 
