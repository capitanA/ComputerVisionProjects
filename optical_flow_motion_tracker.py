import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import os

global numCorners
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def optical_flow(old_points, old_gray):
    global count
    count = 0
    color = np.random.randint(0, 255, (100, 3))
    mask = np.zeros_like(old_frame)
    while cap.isOpened():

        retval, new_frame = cap.read()
        if not retval:
            break
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        count = +1
        new_points, status, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)

        # selecting  good points
        new_points = new_points[status == 1]
        old_points = old_points[status == 1]
        for i, (old, new) in enumerate(zip(old_points, new_points)):
            a, b = old.ravel()
            c, d = new.ravel()
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2, cv2.LINE_AA)
            cv2.circle(new_frame, (int(a), int(b)), 2, color[i].tolist(), -1)

        # display every 5th frame
        display_frame = cv2.add(new_frame, mask)
        out.write(display_frame)
        if count % 5 == 0:
            plt.imshow(display_frame[:, :, ::-1])
            plt.show()
        if count > 50:
            break
        old_gray = new_gray.copy()
        old_points = new_points.reshape(-1, 1, 2)


def corner_detector(frame):
    numCorners = 100
    feature_params = dict(maxCorners=numCorners,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    corners = cv2.goodFeaturesToTrack(frame, **feature_params)
    return corners


if __name__ == "__main__":
    curent_path = os.getcwd()
    cap = cv2.VideoCapture(curent_path + "/videos/cycle.mp4")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    # out = cv2.VideoWriter("sparc_out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20, (width, height))
    out = cv2.VideoWriter('results/sparse-output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (int(width), int(height)))

    success, old_frame = cap.read()
    if success:
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_points = corner_detector(old_gray)
        optical_flow(old_points, old_gray)
