import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import os


def transform_builder(prev_frame, cap, frame_n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_n - 2):
        # finding good feature for each frame
        success, next_frame = cap.read()
        if success != True:
            break
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        old_points = cv2.goodFeaturesToTrack(prev_frame, max_corner, 0.01, 30, blockSize=3)
        new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, old_points, None)

        idx = np.where(status == 1)[0]
        good_old_points = old_points[idx]
        good_new_points = new_points[idx]

        m = cv2.estimateAffinePartial2D(old_points, new_points, )
        ### finding the translation components:Dx,Dy
        dx = m[0][0, 2]
        dy = m[0][1, 2]
        ###finding the rotation component:theta
        theta = np.arctan2(m[0][1, 0], m[0][0, 0])
        transform[i] = [dx, dy, theta]
        prev_frame = next_frame
    return transform


def moving_average(curve, radius):
    kernel_size = 2 * radius + 1
    filter = np.ones(kernel_size) / kernel_size
    paded_trajectory = np.lib.pad(curve, (radius, radius), "edge")
    smoothed_curve = np.convolve(paded_trajectory, filter, mode='same')
    smoothed_curve = smoothed_curve[radius:-radius]
    return smoothed_curve


def curve_smother(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], smoother_radios)
    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def perform_transformm_to_out(cap, transforms_smooth):
    for i in range(frame_n - 2):
        success, frame = cap.read()
        if not success:
            break
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        theta = transforms_smooth[i, 2]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(theta)
        m[0, 1] = -np.sin(theta)
        m[1, 0] = np.sin(theta)
        m[1, 1] = np.cos(theta)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        frame_stabilized = fixBorder(frame_stabilized)
        res_frame = cv2.hconcat([frame, frame_stabilized])
        if (res_frame.shape[1] > 1920):
            res_frame = cv2.resize(res_frame, (width, height))

        out_vid.write(res_frame)


if __name__ == "__main__":
    """Setting som parameters"""
    max_corner = 200
    smoother_radios = 50
    cap = cv2.VideoCapture("videos/pianist.mp4")
    frame_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_vid = cv2.VideoWriter("results/stabilized_video.avi", cv2.VideoWriter_fourcc("M", "j", "P", "G"), fps, (width * 2, height))
    transform = np.zeros((frame_n - 1, 3))
    stat, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    transform_matrix = transform_builder(prev_frame_gray, cap, frame_n)
    trajectory = np.cumsum(transform_matrix, axis=0)
    smoothed_trajectory = curve_smother(trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transform_matrix + difference
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    perform_transformm_to_out(cap, transforms_smooth)
    cv2.destroyAllWindows()
    out_vid.release()
