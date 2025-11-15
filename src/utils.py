import os
import cv2

def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Face detector
    faceProto = os.path.join(MODEL_DIR, "deploy.prototxt")
    faceModel = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    # Age model
    ageProto = os.path.join(MODEL_DIR, "deploy_age.prototxt")
    ageModel = os.path.join(MODEL_DIR, "age_net.caffemodel")

    # Gender model
    genderProto = os.path.join(MODEL_DIR, "deploy_gender.prototxt")
    genderModel = os.path.join(MODEL_DIR, "gender_net.caffemodel")

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    return faceNet, ageNet, genderNet
