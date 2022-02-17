import random
import ipdb
import cv2
import numpy as plt
import matplotlib.pyplot as plt
import os.path

"""In this script 8 different tracking algorithms exist which implemented by opencv"""
"""

   First, User can first choose which object you would like to be track and then a video spcified to that object will be chosen.
   Second, the user define which algorithm they want to use.
   Finally, the output video will be saved in a directory called results.

"""

TRACKING_OBJECTS = dict({"cyclist": "cycle.mp4"})
BBOXES = dict({"cyclist": [(471, 250, 66, 159), (349, 232, 69, 102)]})
TRACKING_ALGORITHMS = list(['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE'])


def draw_rectangles(bboxes, frame):
    for index, bbox in enumerate(bboxes):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, colors[index], 2, cv2.LINE_AA)
        out_cap.write(frame)


def createTrackerByName(algorithm):
    if algorithm == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif algorithm == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif algorithm == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif algorithm == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif algorithm == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif algorithm == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif algorithm == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerMOSSE_create()
    return tracker


def doTrack(first_frame, tracking_object):
    multitracker = cv2.MultiTracker_create()
    # if algorithm == 'BOOSTING':
    #     tracker = cv2.TrackerBoosting_create()
    # elif algorithm == 'MIL':
    #     tracker = cv2.TrackerMIL_create()
    # elif algorithm == 'KCF':
    #     tracker = cv2.TrackerKCF_create()
    # elif algorithm == 'TLD':
    #     tracker = cv2.TrackerTLD_create()
    # elif algorithm == 'MEDIANFLOW':
    #     tracker = cv2.TrackerMedianFlow_create()
    # elif algorithm == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    # elif algorithm == "CSRT":
    #     tracker = cv2.TrackerCSRT_create()
    # else:
    #     tracker = cv2.TrackerMOSSE_create()
    bboxes = BBOXES[tracking_object]
    # creating the multitracker object
    # draw_rectangles(bboxes, first_frame)
    # plt.imshow(first_frame[:,:,::-1])
    # plt.show()
    for bbox in bboxes:
        multitracker.add(createTrackerByName(algorithm), first_frame, bbox)
    out_cap.write(first_frame)

    count = 0
    while True:
        success, new_frame = cap.read()

        if not success:
            break
        ok, new_bboxes = multitracker.update(new_frame)
        if not ok:
            cv2.putText(new_frame, "Tracking failure detected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[0], 2)
        draw_rectangles(new_bboxes, new_frame)
        count += 1


def user_interaction():
    Tracking_video_name = ""
    tracking_object = input("which object you would like to track ?!Only cyclist is Available at this time!")
    while tracking_object not in TRACKING_OBJECTS:
        print("your input is not specified in the list, Try Again!")
        tracking_object = input("which object you would like to track ?Only cyclist is Available at this time!")
    Tracking_video_name = TRACKING_OBJECTS[tracking_object]
    index_method = input(
        "which algorithms you want your object get tracked by?(please indicate by its place number:\n"
        "'BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE' ")

    while index_method not in str([1, 2, 3, 4, 5, 6, 7, 8]):
        print("oops,your input value was not accepted!")
        index_method = input("please provide a number between 1 to 8 to specify the algorithm!")
    algorithm = TRACKING_ALGORITHMS[int(index_method) - 1]
    return tracking_object, Tracking_video_name, algorithm


if __name__ == "__main__":

    colors = []
    for i in range(3):
        # Select some random colors
        colors.append((random.randint(64, 255), random.randint(64, 255),
                       random.randint(64, 255)))
    current_dir = os.getcwd()
    tracking_object, Tracking_video_name, algorithm = user_interaction()

    cap = cv2.VideoCapture(current_dir + "/videos/" + Tracking_video_name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_cap = cv2.VideoWriter("results/" + tracking_object + ".avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps,
                              (width, height))

    if not cap.isOpened():
        print("the video couldn't be opened!")
    success, first_frame = cap.read()

    if not success:
        print("Cannot read the video file!")

    doTrack(first_frame, tracking_object)
