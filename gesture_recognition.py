import time
import cv2
import copy
from PIL import Image
import cvzone
from cvzone .SelfiSegmentationModule import SelfiSegmentation

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.transforms import transforms
from torch.autograd import Variable

classes = ['backwards', 'downwards', 'forwards', 'left', 'right', 'upwards']


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (64,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # normalisation
        # Shape= (64,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (64,12,150,150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (64,12,75,75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (64,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (64,20,75,75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (64,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (64,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (64,32,75,75)

        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

        # Feed forward function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output


checkpoint = torch.load('C:/Users/joshu/OneDrive/A levels/Other/Crest/Crest code/best_checkpoint.pth')
model = ConvNet(num_classes=6)
model.load_state_dict(checkpoint)
model.eval()

prediction = ''
score = 0


# prediction function
def prediction_func(image, transformer):
    image_tensor = transformer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.numpy().argmax()

    pred = classes[index]

    output = func.softmax(output, dim=1)
    max_output, prediction_output = torch.max(output, dim=1)
    max_output = max_output.detach()
    max_output = max_output.numpy() * 100
    max_output = max_output[0]

    return max_output, pred


# Transforms
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
    transforms.Normalize((0.5), (0.5))
])

def change_res(width, height):
    cam.set(3, width)
    cam.set(4, height)


cam = cv2.VideoCapture(0)
change_res(640, 480)


# parameters
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
# lets the camera warmup
time.sleep(0.1)

segmentor = SelfiSegmentation()


while True:
    # capture frames from the camera
    ret, frame = cam.read()

    frame1 = segmentor.removeBG(frame,(255,255,255),threshold=0.6)
    #test with and without segmentor
    #test threshold

    frame = cv2.bilateralFilter(frame1, 5, 50, 100)
    frame = cv2.flip(frame, 1)

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayImage, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    opencv_image=cv2.cvtColor(thresh, cv2.COLOR_RGB2BGR)
    color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted)
    score, prediction = prediction_func(pil_image,transformer)

    text_thresh = copy.deepcopy(thresh)

    #if prediction == "upwards":
    #    
    #elif prediction == "downwards":
    # 
    #elif prediction == "left":
    #
    #elif prediction == "right":
    #
    #elif prediction == "forwards":
    #
    #else:
    #
    #Add your method of controlling the drone in the selection statement above 
    #   • To control your drone using MAVLink use a Radio Telemetry Kit and connect 
    #     the drone to your ground control software (GCS). Then setup a (virtual) controller
    #     on your GCS and use this python script and the xbox360controller or pyvjoy library
    #     to emulate the appopiate joystick movements to tell the drone how to move. The GCS 
    #     will receive these inputs and send commands to the drone via MAVLink.
    
    #   • To control the DJI Tello use the djitellopy library
    #   • To control your drone using UDP and an ESP2866 use the esptool library
    

    cv2.putText(img = text_thresh,
                text = f"Prediction: {prediction} Certainty: {int(score)}",
                org =  (50, 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color = (0, 0, 255),
                thickness=1)

    cv2.imshow('black and white', text_thresh)
    cv2.imshow('frame1', frame1)

    # if the `c` key is pressed, it will break from the loop
    if cv2.waitKey(1) == ord('c'):
        break


cam.release()
cv2.destroyAllWindows()

