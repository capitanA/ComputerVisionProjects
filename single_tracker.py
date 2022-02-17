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


TRACKING_OBJECTS = dict(
    {"ship": "drone-ship.mp4", "face": "face1.mp4", "hockeyman": "hockey.mp4", "ball": "spinning.mp4",
     "surfer": "surfing.mp4", "car": "car.mp4", "manager": "meeting.mp4"})
BBOXES = dict(
    {"ship": (751, 146, 51, 78), "hockeyman": (129, 47, 74, 85), "face": (237, 145, 74, 88), "car": (71, 457, 254, 188),
     "surfer": (97, 329, 118, 293), "ball": (232, 218, 377, 377)})
TRACKING_ALGORITHMS = list(['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE'])
RED = (0, 0, 255)
BLUE = (255, 128, 0)


def doTrack(frame,tracking_object):
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter("results/"+tracking_object+".avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (width, height))
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
    bbox = BBOXES[tracking_object]
    ok = tracker.init(frame, bbox)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, BLUE, 2, 1)
    out_cap.write(frame)
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        ok, bbox = tracker.update(frame)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, BLUE, 2, 1)

        else:
            cv2.putText(frame, "Tracking failure detected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
        out_cap.write(frame)
        plt.imshow(frame[:, :, ::-1])
        count += 1


plt.show()
print("The result will be saved in a folder named result!")


def user_interaction():
    Tracking_video_name = ""
    tracking_object = input("which object you would like to track ?!ship, face, hockeyman, ball, surfacer,car:")
    while tracking_object not in TRACKING_OBJECTS:
        print("your input is not specified in the list, Try Again!")
        tracking_object = input("which object you would like to track ?!ship, face, hockeyman, ball, surfacer,car:")
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
    current_dir = os.getcwd()
    tracking_object, Tracking_video_name, algorithm = user_interaction()
    cap = cv2.VideoCapture(current_dir + "/videos/" + Tracking_video_name)
    if not cap.isOpened():
        print("the video couldn't be opened!")
    success, frame = cap.read()

    if not success:
        print("Cannot read the video file!")

    doTrack(frame,tracking_object)
