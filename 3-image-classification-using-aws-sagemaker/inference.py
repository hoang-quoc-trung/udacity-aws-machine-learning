import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


def Net():
    """
    Load the pre-trained ResNet18 model, freeze the layers, and add custom layers.
    This model is fine-tuned for a classification task with 133 classes.
    """
    model = models.resnet18(pretrained=True) 
    
    # Freeze all layers in the original CNN part of the model
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 133),
    )
    
    return model


def model_fn(model_dir):
    """
    Load the model checkpoint from the specified directory.
    The model is loaded onto the available device (GPU/CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    
    checkpoint_path = os.path.join(model_dir, "model.pth")
    with open(checkpoint_path, "rb") as f:
        logger.info("Loading the pre-trained dog classifier model...")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info('Model successfully loaded')
    
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    """
    Deserialize the input data (image or URL) based on the specified content type.
    """
    logger.info('Deserializing input data.')
    logger.debug(f'Request body CONTENT-TYPE: {content_type}')
    logger.debug(f'Request body TYPE: {type(request_body)}')

    if content_type == JPEG_CONTENT_TYPE:
        logger.debug('Processing image in JPEG format')
        return Image.open(io.BytesIO(request_body))
    
    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f'Request body received: {request_body}')
        request_data = json.loads(request_body)
        logger.debug(f'Parsed JSON: {request_data}')
        url = request_data['url']
        image_content = requests.get(url).content
        return Image.open(io.BytesIO(image_content))
    
    raise Exception(f'Unsupported ContentType: {content_type}')



def predict_fn(input_image, model):
    """
    Transform the input image and pass it through the model to get predictions.
    """
    logger.info('Running prediction.')

    transformation_pipeline = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    logger.info("Transforming input image")
    transformed_image = transformation_pipeline(input_image)
    transformed_image = transformed_image.unsqueeze(0)

    with torch.no_grad():
        logger.info("Predicting class with the model")
        prediction = model(transformed_image) 
    
    return prediction