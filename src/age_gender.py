import cv2
import argparse
from utils import load_models

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Hombre', 'Mujer']

def detect_age_gender(image_path):

    faceNet, ageNet, genderNet = load_models()

    frame = cv2.imread(image_path)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * \
            (w, h, w, h)
        x1, y1, x2, y2 = box.astype(int)

        face = frame[y1:y2, x1:x2]

        blob2 = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        genderNet.setInput(blob2)
        gender = GENDER_LIST[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob2)
        age = AGE_LIST[ageNet.forward()[0].argmax()]

        text = f"{gender} {age}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.rectangle(frame, (x1, y1),
                      (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Resultado", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Ruta de imagen")
    args = parser.parse_args()

    detect_age_gender(args.input)
