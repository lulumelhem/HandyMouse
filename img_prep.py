import cv2
import numpy as np
from matplotlib import pyplot as plt

background = None

# region of interest (ROI) coordinates
top, right, bottom, left = 100, 400, 600, 700

#init weight for running avg
alpha = 0.5

background = cv2.imread("images/background_1.png", 0)
background = background.astype("float")

def gesture_to_name(ges):
    switcher = {
        1: "up",
        2: "left",
        3: "down",
        4: "right",
    }
    return switcher.get(ges, "invalid")

## Sets the backgoround
def run_avg(image, alpha):
    global background

    # compute weighted avg accumulate it and update the background
    cv2.accumulateWeighted(image, background, alpha)

# Segmenting the hand region
def segment(image, threshold=25):
    global background

    ##find the abs diff between the background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

##########
## Main ##
##########

if __name__ == "__main__":

    count = 0
    ges = 1

    #initialize num of frames
    num_frames = 0

    #keep looping, until interrupted
    # while(1):
    # ges_img = cv2.imread("images/" + gesture_to_name(ges) + "_" + str(count + 1) + ".png", 0)
    # plt.subplot(2, 1, 1), plt.imshow(ges_img, cmap='gray')
    # plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 1, 2), plt.hist(ges_img.ravel(), 256)
    # plt.title('Histogram'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()

    while(1):
        if count < 100:
            img = cv2.imread("images/" + gesture_to_name(ges) + "_" + str(count + 1) + ".png", 0)
            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            plt.imshow(imgf, cmap='gray')
            plt.show()
            count += 1
        else:
            count = 0
            ges += 1

        if ges == 5:
            break

    # #####Get Background
    #     if num_frames < 29:
    #         img = cv2.imread("images/background_" + str(num_frames + 1) + ".png", 0)
    #         run_avg(img, alpha)
    #
    #     else:
    #         if count < 100:
    #             ges_img = cv2.imread("images/" + gesture_to_name(ges) + "_" + str(count+1) + ".png", 0)
    #             hand = segment(ges_img)
    #
    #         # check whether hand region is segmented
    #             if hand is not None:
    #                 (thresholded, segmented) = hand
    #
    #                 cv2.imshow("Thresholded", thresholded)
    #                 cv2.imwrite("pp_imgs/" + gesture_to_name(ges) + "_" + str(count+1) + ".png", thresholded)
    #                 cv2.waitKey(2)
    #                 count += 1
    #         else:
    #             count = 0
    #             ges += 1
    #
    #     if ges == 5:
    #         break
    #
    #     # increment the number of frames
    #     num_frames += 1
    #
    #      # observe the keypress by the user
    #     keypress = cv2.waitKey(1) & 0xFF
    #
    #     # if the user pressed "q", then stop looping
    #     if keypress == ord("q"):
    #         break
    #
    # # free up memory
    # cv2.destroyAllWindows()
    # cv2.waitKey(0)