"""
Test json:
{
  "image_data": "",
  "s3_bucket": "udacity-project-3-trunghoang",
  "s3_key": "final-project/test/bicycle_s_000513.png"
}
"""


# ----------------- Lambda Function 1: Serialize Image Data -----------------

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(bucket, key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# ----------------- Lambda Function 2: Image Classification -----------------


import json
import base64
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer


# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-09-11-09-02-08-226"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    predictor = Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    inferences = inferences.decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': {
            "image_data": event["body"]["image_data"],
            "s3_bucket": event["body"]["s3_bucket"],
            "s3_key": event["body"]["s3_key"],
            "inferences": inferences
        }
    }


# ----------------- Lambda Function 3: Filter-----------------


import json
import ast


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['body']['inferences']
    inferences = ast.literal_eval(inferences)
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any (x > THRESHOLD for x in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
