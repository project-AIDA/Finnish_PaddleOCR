from fastapi import FastAPI, File, UploadFile, HTTPException
import logging
import sys
from order import OrderPolygons
from paddleocr import PaddleOCR
from PIL import Image
import io
import numpy as np
import cv2

# For logging options see
# https://docs.python.org/3/library/logging.html
logging.basicConfig(filename='api_log_gpu.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

#initialize ordering
order = OrderPolygons()

#path to folder where the inference model is located.
model_path = './model/'

try:
    # Initialize API Server
    app = FastAPI()
except Exception as e:
    logging.error('Failed to start the API server: %s' % e)
    sys.exit(1)

# Function is run (only) before the application starts

@app.on_event("startup")
async def load_model():
    """
    Load the pretrained model on startup.
    """
    try:
        #load model 
        model = PaddleOCR(lang='latin', show_log=False, det=True, 
                        use_angle_cls=True,
                        rec_model_dir=model_path, 
                        use_gpu=True)
        # Add model to app state
        app.package = {"model": model}
    except Exception as e:
        logging.error('Failed to load the model file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to load the model file: %s' % e)
    
def predict(path, use_angle_cls, reorder_texts):
    """
    Perform prediction on input image.
    """
    # Get model from app state
    model = app.package["model"]
    res = model.ocr(path, cls=use_angle_cls)
    if reorder_texts:
        boxes = [i[0] for i in res[0]]
        new_boxes = [[box[1][0], box[0][0], box[2][1], box[0][1]] for box in boxes]
        new_order = order.order(new_boxes)
        res[0] = [res[0][i] for i in new_order]
        
    return res[0]

# Endpoint for GET requests: input image path is received with the http request
@app.get("/predict_path")
async def predict_path(path: str, use_angle_cls: bool, reorder_texts: bool):
    # Get predicted class and confidence
    try:
        predictions = predict(path, 
                              use_angle_cls=use_angle_cls, 
                              reorder_texts=reorder_texts)
    except Exception as e:
        logging.error('Failed to analyze the input image file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to analyze the input image file: %s' % e)

    return predictions

# Endpoint for POST requests: input image is received with the http request
@app.post("/predict_image")
async def predict_image(use_angle_cls: bool, reorder_texts: bool, file: UploadFile = File(...)):
    try:
        # Loads the image sent with the POST request
        req_content = await file.read()
        image = np.array(Image.open(io.BytesIO(req_content)).convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error('Failed to load the input image file: %s' % e) 
        raise HTTPException(status_code=400, detail='Failed to load the input image file: %s' % e)

    # Get predicted class and confidence
    try: 
        predictions = predict(image,
                              use_angle_cls=use_angle_cls, 
                              reorder_texts=reorder_texts)
    except Exception as e:
        logging.error('Failed to analyze the input image file: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to analyze the input image file: %s' % e)
        
    return predictions