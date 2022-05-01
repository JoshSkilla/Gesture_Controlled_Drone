# Gesture_Controlled_Drone
The goal of this project was to design a drone system capable of being commanded by aircraft marshalling signals through the use of image-processing and deep learning. The system predominantly utilizes OpenCV and PyTorch libraries for real-time detection and classification. 

The image recognition device emulates a virtual controller on a ground control station to send commands to the drone's flight controller via MAVLink.

data_capture.py -> Program that captured data to train the convolutional neural network (CNN)
testing_model.ipynb -> Notebook that defined the CNN and trained the model
best_checkpoint.pth -> Gesture recognition model
testing_model.ipynb -> Notebook that tested the gesture recognition model against prediction data
gesture_recognition.py -> Program that implements the model in real-time and displays camera frames with live prediction in a popup window

For more information on the drone, programs or progression of this project refer to the MS Word file
