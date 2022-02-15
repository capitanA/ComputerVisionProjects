import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import os

"""There two functions in this script one for face detection and one for smile detection within a face detection using haarcascade feature descripto"""


def face_detector(query_im):
    global faces
    gray_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2GRAY)

    minneighbor = 10
    step_neighbor = 1
    fig = plt.figure(figsize=(20, 8))
    face_detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    for neighbor in range(1, minneighbor, step_neighbor):

        faces = face_detector.detectMultiScale(gray_im, 1.2, neighbor)

        im = query_im.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y),
                          (x + w, y + h),
                          (255, 0, 0), 2)
            cv2.putText(im,
                        "# Neighbors = {}".format(neighbor), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

        fig.add_subplot(3, 3, neighbor)
        plt.imshow(im[:,:,::-1])
    plt.show()


def face_smile_detector(query_im, original_im):
    global face_frame
    minneighbor = 90
    step_neighbor = 10
    count = 1
    fig = plt.figure(figsize=(20, 8))
    face_detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    smile_detector = cv2.CascadeClassifier("models/haarcascade_smile.xml")
    faces = face_detector.detectMultiScale(query_im, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(original_im, (x, y),
                      (x + w, y + h),
                      (255, 0, 0), 2)

        gray_face_frame = query_im[y:y + h, x:x + w]

    for neighbor in range(1, minneighbor, step_neighbor):

        smiles = smile_detector.detectMultiScale(gray_face_frame, 1.5, neighbor)
        original = original_im.copy()
        face_frame = original[y:y + h, x:x + w]
        for (xx, yy, ww, hh) in smiles:
            cv2.rectangle(face_frame, (xx, yy),
                          (xx + ww, yy + hh),
                          (0, 255, 0), 2)

            cv2.putText(original,
                        "# Neighbors = {}".format(neighbor), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

        fig.add_subplot(3, 3, count)
        plt.imshow(original[:, :, ::-1])
        count += 1
    plt.show()


if __name__ == "__main__":
    original_im = cv2.imread("images/hillary_clinton.jpg", 1)
    gray_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2GRAY)

    faces = face_detector(original_im)


    """
    now I want to group two classifier, one for face and one for smile and pass it to Cascade classifier class
    """
    # face_smile_detector(gray_im, original_im)
