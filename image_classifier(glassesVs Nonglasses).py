import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import os

class_name = {"without_glasses": 0, "with_glasses": 1}

train_data = []
test_data = []
train_label = []
test_label = []
current_directory = os.getcwd()

"""Tis script is for image classification (with sunglasses faces Vs without sunglasses faces) """
#########################################################
###########    Initialize hog parameters   ##############
#########################################################
winSize = (96, 32)
blockSize = (8, 8)
blockStride = (8, 8)
cellSize = (4, 4)
nbins = 9
derivAperture = 0
winSigma = 4.0
histogramNormType = 1
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 1
nlevels = 64

#########################################################
###########    Creating the HOG Object   ##############
#########################################################
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                        nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold,
                        gammaCorrection, nlevels, 1)


def get_train(label, test_fraction=0.2):
    if label == 1:
        class_name = "/cropped_withGlasses2"
        image_path = current_directory+"/datasets" + class_name

    else:
        class_name = "/cropped_withoutGlasses2"
        image_path = current_directory +"/datasets" + class_name
    images_name = os.listdir(image_path)
    n_test = int(len(images_name) * test_fraction)
    for index, image_name in enumerate(images_name):
        img = cv2.imread(image_path + "/" + image_name, 1)
        if index <= n_test:
            test_data.append(img)
            test_label.append(label)
        else:
            train_data.append(img)
            train_label.append(label)

    return train_data, train_label, test_data, test_label


def svm_init(c, gamma):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(c)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model


def svm_train(model, samples, response):
    model.train(samples, cv2.ml.ROW_SAMPLE, response)
    return model


def svm_predict(model, sample):
    return model.predict(sample)[1].ravel()


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
    return hogdata



if __name__ == "__main__":
    #########################################################################
    ###########    get the data and load them into SVM for training   ##############
    #########################################################################

    train_data_glasses, train_label_glasses, test_data_glasses, test_label_glasses = get_train(
        class_name["without_glasses"], 0.2)
    train_data_withoutglasses, train_label_withoutglasses, test_data_withoutglasses, test_label_withoutglasses = get_train(
        class_name["with_glasses"], 0.2)

    ####################################################################################################################
    ###########    the data needs to be concatenated to each others and then feed to the svm for training   #############
    ####################################################################################################################
    training_data = np.concatenate((np.array(train_data_glasses), np.array(train_data_withoutglasses)), axis=0)
    training_label = np.concatenate((np.array(train_label_glasses), np.array(train_label_withoutglasses)), axis=0)

    testing_data = np.concatenate((np.array(test_data_glasses), np.array(test_data_withoutglasses)), axis=0)
    testing_label = np.concatenate((np.array(test_label_glasses), np.array(test_label_withoutglasses)), axis=0)

    #########################################################################################################################
    ###########    Now we need to pass this dat to hod descriptor to produce the feature vectors for our data   #############
    #########################################################################################################################
    train_hog = compute_hog(hog, training_data)
    test_hog = compute_hog(hog, testing_data)

    train_features = prepareData(train_hog)
    test_features = prepareData(test_hog)

    model = svm_init(c=2.5, gamma=0.02)
    model = svm_train(model=model, samples=train_features, response=training_label)
    evaluate(model, test_features, testing_label)




    
