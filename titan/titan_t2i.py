# docs: 
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

import base64
import io
import json
import boto3
from PIL import Image

br = boto3.client(service_name='bedrock-runtime')

AVAILABLE_MODEL_IDS = [
    'amazon.titan-image-generator-v1'
]
model_id = AVAILABLE_MODEL_IDS[-1]

prompt = 'A rising Chinese dragon painted by an Impressionist artist'
negative_prompt = 'deformed, disfigured, extra limbs, poorly drawn, bad anatomy, blurry, text, watermark, signature, low quality'
body = json.dumps({
    'taskType': 'TEXT_IMAGE',
    'textToImageParams': {
        'text': prompt,
        'negativeText': negative_prompt
    },
    'imageGenerationConfig': {
        'numberOfImages': 2,
        'height': 1024,
        'width': 1024,
        'cfgScale': 8.0,
        'seed': 42
    }
})

accept = 'application/json'
content_type = 'application/json'

response = br.invoke_model(
    body=body, modelId=model_id, accept=accept, contentType=content_type
)
response_body = json.loads(response.get('body').read())
base64_images = response_body.get('images')
for i, b64_image in enumerate(base64_images):
    b64_bytes = b64_image.encode('ascii')
    image_bytes = base64.b64decode(b64_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    image.show()
    image.save(f'{str(i).zfill(3)}.png')