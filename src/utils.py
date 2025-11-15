import cv2

def load_models():
    face_model = cv2.dnn.readNet(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )

    age_model = cv2.dnn.readNet(
        "models/deploy_age.prototxt",
        "models/age_net.caffemodel"
    )

    gender_model = cv2.dnn.readNet(
        "models/deploy_gender.prototxt",
        "models/gender_net.caffemodel"
    )

    return face_model, age_model, gender_model
