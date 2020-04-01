import cv2
import tensorflow as tf
from tensorflow import keras
import imutils          # imutils -> functions to make basic image processing functions
import numpy as np      # Numpy -> package for scientific computing


# Goal: to segment the image by subtracting the background
# Will use running averages to do this

background = None;

####################
## Get Background ##
####################
def getBackground (image, alpha):
    global background
    if background is None:
        background = image.copy().astype("float")
        return

##################
## Segmentation ##
##################
def segment (image):
    diff = cv2.absdiff(background.astype("uint8"), image)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # return if no contours found
    if len(contours) == 0:
        return
    # else:
    #     segmented = max(contours, key=cv2.contourArea)
    return (thresh, contours)


###################
## Draw Contours ##
###################
def drawContours(image, contours):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    hull = cv2.convexHull(sorted_contours[0])
    # defects = cv2.convexityDefects(sorted_contours[0], hull)
    cv2.drawContours(image, [hull + (right, top)], -1, (0, 255, 0), 3)
    # cv2.drawContours(image, [defects + (right, top)], -1, (0, 255, 0), 3)

##########
## Main ##
##########

if __name__ == "__main__":

    model = keras.models.load_model('hg_trained_model.h5')
    model.summary()

    # region of interest (ROI) coordinates
    # This is the frame of interest
    # top, right, bottom, left = 100, 200, 600, 700
    top, right, bottom, left = 150, 400, 600, 700

    #get the ref to the webcam
    webcam = cv2.VideoCapture(2)

    #initialize num of frames
    num_frames = 0

    #keep looping, until interrupted
    while(1):

        # get the current frame
        (grabbed, frame) = webcam.read()

        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        # get the ROI
        roi = frame[top:bottom, right:left]
        # convert the roi to grayscale and blur it
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Image smoothing -> makes img less pixelated (removes noise)
        blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)  # Uses Gaussian Kernel to blur the images

        if num_frames < 30:
            getBackground(blur, alpha=0.5)
        else:
            hand = segment(blur)

            if hand is not None:
                #unpack the segmented hand image
                (thresh, contours) = hand

                #Draw the segmented region and display the hand
                drawContours(frame_copy, contours)
                cv2.imshow("Thresh", thresh)

                thresh.resize(1, 32, 32, 1, refcheck=False)
                thresh_test = np.array(thresh, dtype="uint8")
                # thresh_test = thresh_test/255
                print(thresh_test.shape)
                result = model.predict_classes(thresh_test)

        cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Clone", frame_copy)

        num_frames += 1
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
    webcam.release()
    cv2.destroyAllWindows()





