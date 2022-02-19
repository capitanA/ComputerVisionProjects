import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipdb


def run_dnn():
    net = cv2.dnn.readNetFromDarknet(yolo_config_filename, yolo_weights_filename)
    outlayers = getoutputsNames(net)
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=True)
    net.setInput(blob)
    outs = net.forward(outlayers)
    postprocess(input_image, outs)
    plt.imshow(input_image[:, :, ::-1])
    plt.show()


def getoutputsNames(net):
    # we search over all the unconnected output layesr with net.getUnconnectedOutLayers() and then by having their index get their names from layaersNames
    layaersNames = net.getLayerNames()
    return [layaersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.

    classIds = []
    confidences = []
    boxes = []
    for indexouts, out in enumerate(outs):
        for index, detection in enumerate(out):
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - (width / 2))
                    top = int(center_y - (height / 2))
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(input_image, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classnames:
        assert (classId < len(classnames))
        label = '%s:%s' % (classnames[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(input_image, (left, top - round(1.5 * labelSize[1])),
                  (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(input_image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


if __name__ == "__main__":
    inWidth = 416
    inHeight = 416
    objectnessThreshold = 0.5  # Objectness threshold
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416  # Height of network's input image
    yolo_config_filename = "DNN_models/yolov3.cfg"
    yolo_weights_filename = "DNN_models/yolov3.weights"
    classnames_files = "DNN_models/coco.names"
    input_image = cv2.imread("images/bird.jpg")

    with open(classnames_files, "r") as f:
        classnames = f.read().split("\n")
    run_dnn()
