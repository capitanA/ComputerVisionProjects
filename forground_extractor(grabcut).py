import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import os

"""
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up,  one for input and one for output.

    At first,  in input window,  draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups,  you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
Key 'Esc'- To Exit
===============================================================================
"""

# Convention of defining color in opencv is BGR
LIGHT_GREEN = [128, 255, 128]  # rectangle color
LIGHT_RED = [128, 128, 255]  # PR BG
BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_BG = {"color": RED, "val": 0}
DRAW_FG = {"color": GREEN, "val": 1}
DRAW_PR_FG = {'color': LIGHT_GREEN, 'val': 3}
DRAW_PR_BG = {'color': LIGHT_RED, 'val': 2}

### setting some flags
rect = (0, 0, 1, 1)
VALUE = DRAW_BG
drawing = False  # flag for drawing curves
rectangle = False  # flag for drawing rect
rect_over = False  # flag to check if rect drawn
rect_or_mask = 100  # flag for selecting rect or mask mode
thickness = 3  # brush thickness
rect_not_done = True
draw_circle = False
circle_not_done = True


def onmouse(event, x, y, flags, params):
    global rectangle, rect_not_done, pnt1, pnt2, img, rect_over, draw_circle, circle_not_done, ix, iy, im, rect, rect_or_mask

    if event == cv2.EVENT_LBUTTONDOWN and rect_not_done:
        rectangle = True
        ix, iy = int(x), int(y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle:
            im = img2.copy()
            pnt2 = (int(x), int(y))
            cv2.rectangle(im, (ix, iy), pnt2, (255, 0, 0), 3)
            rect = (min(ix, x), min(iy, y), abs(ix - x),
                    abs(iy - y))  # this  is the coordinates of the drawn rectangle in the image with the format of: (x,y,w,h)
            rect_or_mask = 0
            cv2.imshow("Input", im)
    elif event == cv2.EVENT_LBUTTONUP and rect_not_done:
        rectangle = False
        rect_not_done = False
        rect_over = True
        pnt2 = (int(x), int(y))
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        cv2.rectangle(im, (ix, iy), pnt2, (255, 0, 0), 3)
        cv2.imshow("Input", im)
        rect_or_mask = 0

    ### touchup curves
    if event == cv2.EVENT_LBUTTONDOWN and not rect_not_done:
        if not rect_over:
            print("first draw a bounding box!")
        else:
            draw_circle = True
            center = (int(x), int(y))
            cv2.circle(im, center, 4, VALUE["color"], -1)
            cv2.circle(mask, (int(x), int(y)), 5, VALUE["val"])
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_circle:
            cv2.circle(im, (int(x), int(y)), 4, VALUE["color"], -1)
            cv2.circle(mask, (int(x), int(y)), 5, VALUE["val"])
            cv2.imshow("Input", im)
    elif event == cv2.EVENT_LBUTTONUP and not rect_not_done:
        if draw_circle:
            draw_circle = False
            cv2.circle(im, (int(x), int(y)), 4, VALUE["color"], -1)
            cv2.circle(mask, (int(x), int(y)), 5, VALUE["val"])
            cv2.imshow("Input", im)


if __name__ == "__main__":
    im = cv2.imread("images/messi.jpeg", cv2.IMREAD_COLOR)
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Input", (600, 700))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", (600, 700))
    cv2.setMouseCallback("Input", onmouse)
    cv2.moveWindow('Input', im.shape[1] + 10, 90)
    bgdmodel = np.zeros((1, 65),
                        np.float64)  ## this needs to be always like this.  pass this array to the grab_cut algorithm.
    fgdmodel = np.zeros((1, 65),
                        np.float64)  ## this needs to be always like this.  pass this array to the grab_cut algorithm.

    img2 = im.copy()
    mask = np.zeros(im.shape[:2], dtype=np.uint8)
    output = np.zeros(im.shape, np.uint8)
    while True:
        cv2.imshow("Input", im)
        cv2.imshow("output", output)
        k = cv2.waitKey(0)
        if k == 27:
            break
        if k == ord("s"):  ## this must save the output image
            cv2.imwrite("results/outputimage.jpg", output)

        elif k == ord("r"):  ## this must reset the image
            rect = (0, 0, 1, 1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            rect_not_done = True
            value = DRAW_FG
            im = img2.copy()
            mask = np.zeros(im.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
            output = np.zeros(im.shape, np.uint8)

        elif k == ord("0"):  # this means the user want to specify part of the image which is background for sure
            VALUE = DRAW_BG

        elif k == ord("1"):  # this means the user want to specify part of the image which is foreground for sure
            VALUE = DRAW_FG

        elif k == ord("2"):  # this means the user want to specify part of the image which is probably background
            VALUE = DRAW_PR_BG

        elif k == ord("3"):  # this means the user want to specify part of the image which is probably foreground
            VALUE = DRAW_PR_FG

        elif k == ord("n"):  ## here the GrabCut algorithm need to be revoke!
            if rect_or_mask == 0:  # grabcut with rect
                cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, mode=cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:  # grabcut with mask
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask=mask2)
    cv2.destroyAllWindows()
