# docs: 
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

import boto3
import json
import base64
import os

br = boto3.client(service_name='bedrock-runtime')

ANTHROPIC_VERSION = 'bedrock-2023-05-31'

AVAILABLE_MODEL_IDS = [
    'anthropic.claude-3-haiku-20240307-v1:0','anthropic.claude-3-sonnet-20240229-v1:0', # v3
]

model_id = AVAILABLE_MODEL_IDS[-1] # claude-3-sonnet

def make_image_content(file_path):
    full_path = os.path.join(os.path.dirname(__file__),f'{file_path}')
    ext = full_path.split('.')[-1]
    with open(full_path,'rb') as f:
        content_image = base64.b64encode(f.read()).decode('utf-8')
    return {
        "type": "image", 
                "source": {
                    "type": "base64",
                    "media_type": f"image/{ext}", 
                    "data": content_image
                }
    }

system_prompt = '以下はユーザーと AI の日本語でのやり取りです。AI はユーザーの質問に親切な日本語で返します。'
messages = [
    {
        "role": "user","content": [
            make_image_content('./image/lattice.jpeg'),
            {
                "type": "text", 
                "text": '与えた画像は日本の 2 種類のお菓子です。何がいくつ映っていますか？'
            },
        ]
    },
]

body = json.dumps({
    'anthropic_version': ANTHROPIC_VERSION,
    'max_tokens': 4096,
    'system': system_prompt,
    'messages': messages,
    'temperature': 0,
    'top_p': 0,
    'top_k': 0,
    'stop_sequences': []
})

response = br.invoke_model_with_response_stream(body=body, modelId=model_id)
for event in response.get("body"):
    chunk = json.loads(event["chunk"]["bytes"])

    if chunk['type'] == 'message_delta':
        print(f"\nStop reason: {chunk['delta']['stop_reason']}")
        print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
        print(f"Output tokens: {chunk['usage']['output_tokens']}")

    if chunk['type'] == 'content_block_delta':
        if chunk['delta']['type'] == 'text_delta':
            print(chunk['delta']['text'], end="")
