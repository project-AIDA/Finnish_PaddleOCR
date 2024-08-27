# Finnish_PaddleOCR

This repository contains models developed in project Potential of Artificial Intelligence for Digital Archives' Users that was co-funded by the European Union. 

It is a cooperation work between National Archives of Finland (NAF), Central Archives for Finnish Business (ELKA) and South East University of Finland (Xamk).

The models are finetuned from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_en.md) latin model, using a Finnish dataset from ELKA and combining that with Finnish synthetic data. All the data used in model training can be found from huggingface.

We found that character error rate (CER) is about 3 times smaller than with the base model in our tests ja 2 times smaller compared to Tesseract in Finnish datasets from ELKA and National Archives of Finland.
   
# Installation

It is recommended to use a virtual environment (e.g. conda or venv) for the installation. The installation is tested on python version 3.11. Depending on your hardware the installation is a bit different. On cpu, you can install simply by activating your desired virtual environment and writing the command below.

`pip install -r requirements_cpu.txt`

## GPU installation

For gpu installation, you should first install PaddlePaddle deep learning library based on your CUDA version. You can find different versions to choose from here https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/windows-pip_en.html. After that you can install rest of the required libraries using the following command:

`pip install -r requirements_gpu.txt`

You can find further information on installing PaddleOCR here https://paddlepaddle.github.io/PaddleOCR/en/ppocr/quick_start.html. 

# Usage 

## Code 

Here is an example how you can use our trained model with cpu. 

```
from paddleocr import PaddleOCR

use_angle_classifier = False
onnx_rec_model = './onnx_model/rec_onnx/model.onnx'
onnx_det_model = './onnx_model/det_onnx/model.onnx'
onnx_cls_model = './onnx_model/cls_onnx/model.onnx'

ocr = PaddleOCR(lang='latin', show_log=False, det=True, 
                use_angle_cls=use_angle_classifier,
                rec_model_dir=onnx_rec_model, 
                det_model_dir=onnx_det_model,
                cls_model_dir=onnx_cls_model,
                use_gpu=False,
                use_onnx=True)

res = model.ocr("/path/to/image.jpg", cls=use_angle_classifier)
```

Here is how you can use the model with gpu.


```
from paddleocr import PaddleOCR

use_angle_classifier = False
model_path = './model/'

ocr = PaddleOCR(lang='latin', show_log=False, det=True, 
               use_angle_cls=use_angle_classifier,
               rec_model_dir=model_path, 
               use_gpu=True)

res = model.ocr("/path/to/image.jpg", cls=use_angle_classifier)
```

In both examples, you can set `use_angle_classifier = True`. This will try to detect if the textline is upside down.   

## API

The API code has been built using the [FastAPI](https://fastapi.tiangolo.com/) library. This repository contains 2 APIs for running our trained PaddleOCR, one for using a cpu and another one for using a gpu. The cpu version uses [ONNX](https://onnx.ai/) model format for increased inference speeds. 

You can set up your desired API by using a command `uvicorn api_onnx:app` or `uvicorn api_gpu:app`. These commands set an API into the default port of 8000 in localhost. 

You can change the host and the port as shown below:

`uvicorn api_onnx:app --host 0.0.0.0 --port 8080`

## API methods 

The APIs have a method for predicting the text in the image called `/predict_path`. It is designed to take the path to the image as an input. It takes following arguments as input:

>   path: str (path to the image)
>   
>   use_angle_cls: bool (whether to use angle classifier for text lines)
>   
>   reorder_texts: bool (whether to reorder texts)

Another method is called `/predict_image`. It is designed to take an image in binary form as an input. It takes following arguments as input:

> file: File (a binary file of the image)
>
> use_angle_cls: bool (whether to use angle classifier for text lines)
>   
> reorder_texts: bool (whether to reorder texts)

Both of these methods output a list of textline detections. The detections include the coordinates, recognized text and the confidence value. If PaddleOCR does not recognize anything the output is empty. 

## Calling the APIs

### Curl

You can try calling the APIs with e.g. the `curl` command. 

`curl 'http://localhost:8000/predict_path?path=/path/to/the/image.jpg&use_angle_cls=false&reorder_texts=false'`

`curl 'http://localhost:8000/predict_image?use_angle_cls=false&reorder_texts=false' -F 'file=@/path/to/the/image.jpg'`

NB! Windows users might encounter following error `Invoke-WebRequest : A parameter cannot be found that matches parameter name 'F'.`. This can be bypassed by running a command `Remove-item alias:curl`.

### Python

In python, you can use e.g. requests library to access the API. Here is an example.

```
import requests

path = '/path/to/image.jpg'
cls = False
reorder = False

# Predict image by path
response = requests.get('http://localhost:8000/predict_path?path=' + str(path) +  '&use_angle_cls=' + str(cls) + '&reorder_texts=' + str(reorder))
print(response.json())

# Predict image by image
response = requests.post('http://localhost:8000/predict_image?use_angle_cls=' + str(cls) + '&reorder_texts=' + str(reorder), 
                         files={'file': open(path, 'rb')})
print(response.json())
```

## Logging

Logging events are saved into a file `api_log_onnx.log`/`api_log_gpu.log` in the same folder where the `api_onnx.py`/`api_gpu.py` file is located. Previous content of the log file is overwritten after each restart. More information on different logging options is available [here](https://docs.python.org/3/library/logging.html).

# Results

We have tested the model performance on our own data. The results are shown below. Out of these datasets the only public dataset is the ELKA test set that can be found in huggingface. 

| Model      | ELKA test set | DALAI test set [link to paper](https://ieeexplore.ieee.org/abstract/document/10252214)  | NAF test set
| ----------- | ----------- | ------| - |
| Tesseract      |    0.046    | 0.027 | 0.044
| Original PaddleOCR   |   0.067   | 0.039 | 0.068
| Our PaddleOCR | 0.020 | 0.012 | 0.023

All of the datasets used here are typewritten. Handwritten text was used during the training, but the results were poor. That is why, when using the model for handwritten text, you should use extra caution. 

# Training

In case you want to finetune our trained model, you should refer to the PaddleOCR docs here https://paddlepaddle.github.io/PaddleOCR/en/ppocr/model_train/recognition.html. The training model can be found in huggingface https://huggingface.co/Kansallisarkisto/PaddleOCR_training. Additionally, the training data can be found here https://huggingface.co/datasets/Kansallisarkisto/AIDA_ocr_training_data. 

## Synthetic data

As a way to increase the amount of training data, we created synthetic data by using this library https://github.com/Belval/TextRecognitionDataGenerator. We collected Finnish books from https://www.gutenberg.org/ and Finnish magazines from https://archive.org/ and created different kinds of textlines. The different kinds include normal textlines, rotated textlines, textlines following a sinosoidal curve and textlines where characters are subjected to noise. 
