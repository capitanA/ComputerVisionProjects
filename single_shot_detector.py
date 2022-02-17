import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipdb


def detect_objects(img, net):
    dim = 300
    mean = (127.5, 127.5, 127.5)
    scale = 1
    swapRB = True
    blob = cv2.dnn.blobFromImage(img, scale / 127.5, (dim, dim), mean, swapRB)
    net.setInput(blob)
    objects = net.forward()
    return objects


def display_text(im, text, x, y):
    FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1
    cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


def display_objects(im, objects, threshold=0.25):
    # [Unknown, classId, Confidence,X,Y, Width, HEIGHT] ====> the format for properties stored in the objects OBJECT!

    # First we need to get the original properties of the image
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        if score > threshold:
            display_text(im, f"{classes[classId]}", x, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)


if __name__ == "__main__":
    img = cv2.imread("images/street.jpg")
    config_file = "DNN_models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    model_file = "DNN_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
    classes_file = "DNN_models/coco_class_labels.txt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    classes = None
    with open(classes_file, "rt") as f:
        classes = f.read().split("\n")
    objects = detect_objects(img, net)
    display_objects(img, objects)
    plt.imshow(img)
    plt.show()
