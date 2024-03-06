# docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

import boto3
import json
import base64
import os

br = boto3.client(service_name='bedrock-runtime')

ANTHROPIC_VERSION = 'bedrock-2023-05-31'

AVAILABLE_MODEL_IDS = [
    'anthropic.claude-3-sonnet-20240229-v1:0', # v3
]

model_id = AVAILABLE_MODEL_IDS[-1] # claude-3-sonnet

file_path = os.path.join(os.path.dirname(__file__),'image/img_way-to-draw-architecture_03.08e1366ebaecaf76a333002acbe1e2bd363acd4b.png')
ext = file_path.split('.')[-1]
with open(file_path,'rb') as f:
    content_image = base64.b64encode(f.read()).decode('utf-8')

system_prompt = '以下はユーザーと AI のやり取りです。ユーザーは様々なセンサーの画像を与えます。AI はセンサーの値を読み取ってください。'
messages = [
    {
        "role": "user","content": [
            {
                "type": "image", 
                "source": {
                    "type": "base64",
                    "media_type": f"image/{ext}", 
                    "data": content_image
                }
            },
            {
                "type": "text", "text": '画像は AWS を用いたアーキテクチャーです。説明してください。'
            },
        ]
    },
]

body = json.dumps({
    "anthropic_version": ANTHROPIC_VERSION,
    "max_tokens": 4096,
    "messages": messages,
    'temperature': 0,
    'top_p': 0,
    'top_k': 0,
    'stop_sequences': []
})

response = br.invoke_model(body=body, modelId=model_id)
response_body = json.loads(response.get('body').read())
print(response_body['content'][0]['text'])