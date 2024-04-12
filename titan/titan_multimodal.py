import boto3
import json
import os
import base64
import numpy as np

br = boto3.client(service_name='bedrock-runtime')

def make_image_content(file_path):
    full_path = os.path.join(os.path.dirname(__file__),f'{file_path}')
    ext = full_path.split('.')[-1]
    with open(full_path,'rb') as f:
        content_image = base64.b64encode(f.read()).decode('utf-8')
    return content_image

body=json.dumps({
    "inputText": "dog",
    "inputImage": make_image_content('./image/dog.jpeg'),
    "embeddingConfig": { "outputEmbeddingLength": 1024}
})

response = br.invoke_model(body=body, modelId='amazon.titan-embed-image-v1')
img1 = np.array(json.loads(response['body'].read())['embedding'])

body=json.dumps({
    "inputText": "dog",
    "inputImage": make_image_content('./image/dog2.jpeg'),
    "embeddingConfig": { "outputEmbeddingLength": 1024}
})

response = br.invoke_model(body=body, modelId='amazon.titan-embed-image-v1')
img2 = np.array(json.loads(response['body'].read())['embedding'])

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(cos_sim(img1,img2))