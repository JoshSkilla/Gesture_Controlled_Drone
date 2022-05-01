from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import copy
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.rotation = 270
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
#camera.contrast = 85
#camera.brightness = 55
#best if brightness is done as the image not as config

#parameters
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
save_images = True
selected_gesture = "forwards"
img_counter = 1
# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    image = cv2.bilateralFilter(image, 5, 50, 100)
    image = cv2.flip(image, 1)

    # show the frame
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayImage, (blurValue, blurValue), 0)
    #(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 480, 640, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh1 = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #might have to change around values here for it to work
    #_, contours, hierarchy
    #_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(image.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

    #cv2.imshow('output', drawing)
    cv2.imshow('black and white', thresh)
    #
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("c"):
        break
    else:
        if save_images == True:
            img_name = f"/home/pi/Documents/Dataset/imgRecognition/signal_frames/drawing/{selected_gesture}_{img_counter}.jpg"
            cv2.imwrite(img_name, drawing)
            print("{} written".format(img_name))

            img_name2 = f"/home/pi/Documents/Dataset/imgRecognition/signal_frames/silhouettes/{selected_gesture}_{img_counter}.jpg"
            cv2.imwrite(img_name2, thresh)
            print("{} written".format(img_name2))

            img_counter += 1


