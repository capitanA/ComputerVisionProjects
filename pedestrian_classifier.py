import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import os


def read_dataset():
    # preparing the training data variables
    train_image_pos = []
    train_lbl_pos = []
    train_image_neg = []
    train_lbl_neg = []
    # preparing the testing data variables
    test_image_pos = []
    test_lbl_pos = []
    test_image_neg = []
    test_lbl_neg = []

    if os.path.isdir("INRIAPerson/train_64x128_H96"):
        train_neg_list = os.listdir("INRIAPerson/train_64x128_H96/negPatches")
        train_pos_list = os.listdir("INRIAPerson/train_64x128_H96/posPatches")
        for index, imagename in enumerate(train_pos_list):
            im = cv2.imread("INRIAPerson/train_64x128_H96/posPatches/" + str(imagename))
            train_image_pos.append(im)
            train_lbl_pos.append(1)
        for index, imagename in enumerate(train_neg_list):
            im = cv2.imread("INRIAPerson/train_64x128_H96/negPatches/" + str(imagename))
            train_image_neg.append(im)
            train_lbl_neg.append(-1)

    if os.path.isdir("INRIAPerson/test_64x128_H96"):
        test_neg_list = os.listdir("INRIAPerson/test_64x128_H96/negPatches")
        test_pos_list = os.listdir("INRIAPerson/test_64x128_H96/posPatches")

        for index, imagename in enumerate(test_pos_list):
            im = cv2.imread("INRIAPerson/test_64x128_H96/posPatches/" + str(imagename))
            test_image_pos.append(im)
            test_lbl_pos.append(1)
        for index, imagename in enumerate(test_neg_list):
            im = cv2.imread("INRIAPerson/test_64x128_H96/negPatches/" + str(imagename))
            test_image_neg.append(im)
            test_lbl_neg.append(-1)

    train_data = np.concatenate((np.array(train_image_pos), np.array(train_image_neg)), axis=0)
    train_lbl = np.concatenate((np.array(train_lbl_pos), np.array(train_lbl_neg)), axis=0)

    test_data = np.concatenate((np.array(test_image_pos), np.array(test_image_neg)), axis=0)
    test_lbl = np.concatenate((np.array(test_lbl_pos), np.array(test_lbl_neg)), axis=0)
    return train_data, train_lbl, test_data, test_lbl


def init_svm(c, gamma):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(c)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_EPS +
                           cv2.TERM_CRITERIA_MAX_ITER,
                           1000, 1e-3))
    return model


def train_svm(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svm_predict(model, samples):
    return model.predict(samples)[1]


def evaluate(model, testdata, test_label):
    predictions = svm_predict(model, testdata)
    accuracy = (predictions == test_label).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy * 100))
    return accuracy


def prepareData(data):
    featureVectorLength = len(data[0])
    features = np.float32(data).reshape(-1, featureVectorLength)
    return features


def compute_hog(hog, data):
    hogdata = []
    for image in data:
        hogFeatures = hog.compute(image)
        hogdata.append(hogFeatures)
    print("Done")
    return hogdata


def test_multiscale_detection():
    global queryImage
    saved_model = cv2.ml.SVM_load("pedestrianmodel.yml")
    sv = saved_model.getSupportVectors()
    rho, aplha, svidx = saved_model.getDecisionFunction(0)
    svmDetector = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
    svmDetector[:-1] = -sv[:]
    svmDetector[-1] = rho

    # set our SVMDetector in HOG
    hog.setSVMDetector(svmDetector)

    #######################################################################
    #############    Test our classifier with an new image   ################
    ########################################################################
    images_name = os.listdir("images/pedestrians")

    queryImage = pre_processor(queryImage)

    bboxes, weights = hog.detectMultiScale(queryImage, winStride=(8, 8),
                                           padding=(32, 32), scale=1.05,
                                           finalThreshold=2, hitThreshold=1.0)

    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(queryImage, (x1, y1), (x2, y2),
                      (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    plt.imshow(queryImage[:, :, ::-1])
    plt.show()


def pre_processor(im):
    final_height = 800
    scale = final_height / im.shape[0]
    cv2.resize(im, None, fx=scale, fy=scale)
    return im


def detection_with_opencv_default_svm(hog_obj):
    global queryImage
    default_svm = cv2.HOGDescriptor_getDefaultPeopleDetector()
    hog_obj.setSVMDetector(default_svm)
    queryImage = pre_processor(queryImage)
    bboxes, weights = hog_obj.detectMultiScale(queryImage, winStride=(8, 8),
                                              padding=(32, 32), scale=1.05,
                                              finalThreshold=2, hitThreshold=0.0)
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(queryImage, (x1, y1), (x2, y2),
                      (0, 0, 255), thickness=3,
                      lineType=cv2.LINE_AA)
    plt.imshow(queryImage[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    queryImage = cv2.imread(os.getcwd() + "/images/pedestrians/1.jpg")
    #########################################################
    ###########    Initialize hog parameters   ##############
    #########################################################
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    signedGradient = False

    #########################################################
    ###########    Creating the HOG Object   ##############
    #########################################################
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                            nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels, signedGradient)

    ##########################################################################################
    ###########    this part will train a svm as a classifer for our data set   ##############
    ##########################################################################################
    # train_data, train_lbl, test_data, test_lbl = read_dataset()
    #
    # train_hog = compute_hog(hog, train_data)
    # test_hog = compute_hog(hog, test_data)
    #
    # train_features = prepareData(train_hog)
    # test_features = prepareData(test_hog)
    #
    # model = init_svm(c=0.01, gamma=0.0)
    # model = train_svm(model=model, samples=train_features, responses=train_lbl)
    # model.save("pedestrianmodel.yml")
    # evaluate(model, test_features, test_lbl)

    """ this run a detector in the hog class called multiscaledetector() """
    # test_multiscale_detection()

    """this run a detector in the cv2 library called multiscaledetector()"""
    detection_with_opencv_default_svm(hog)