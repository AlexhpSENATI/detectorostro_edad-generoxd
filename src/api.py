from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from utils import load_models
from age_gender import AGE_LIST, GENDER_LIST

app = FastAPI(title="Age & Gender Detector")

# Cargar modelos una sola vez al iniciar
faceNet, ageNet, genderNet = load_models()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validación simple
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Archivo debe ser una imagen")

    contents = await file.read()
    npimg = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    h, w = frame.shape[:2]

    # Detección de caras
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        # Recortar límites dentro de la imagen
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[y1:y2, x1:x2]

        # Preparar blob para age/gender
        blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                      (78.4263377603, 87.7689143744, 114.895847746),
                                      swapRB=False)

        # Gender
        genderNet.setInput(blob2)
        genderPreds = genderNet.forward()[0]
        genderIdx = int(genderPreds.argmax())
        gender = GENDER_LIST[genderIdx]

        # Age
        ageNet.setInput(blob2)
        agePreds = ageNet.forward()[0]
        ageIdx = int(agePreds.argmax())
        age = AGE_LIST[ageIdx]

        results.append({
            "gender": gender,
            "age": age,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": confidence
        })

    return JSONResponse(content={"results": results})
