import boto3
import json
import os

brt = boto3.client(service_name='bedrock-runtime')
with open('MODEL_IDS.json','rt') as f:
    model_ids = json.load(f)['doc2text']

file_path = 'doc/test-doc.pdf'
file_full_path = os.path.join(os.path.dirname(__file__),f'{file_path}')
file_bin = open(file_full_path, 'rb').read() 

for model_id in model_ids:
    print(f'model_id:{model_id}')
    response = brt.converse(
        # system=[{
        #     'text': ''''''
        # }],
        modelId = model_id,
        messages = [
            {
                'role':'user',
                'content': [
                    {
                        'document': {
                            'name': 'fuga',
                            'format': 'pdf',
                            'source': { 
                                'bytes': file_bin
                            } 
                        }
                    },
                    { 'text': 'fuga の内容を日本語で要約して' }
                ]
            },
        ],
        inferenceConfig={
            'maxTokens':2048,
            'temperature':0,
            # 'stopSequences':[]
        }
    )
    print(response['output']['message']['content'][0]['text'])
    print('\n---------------------\n')