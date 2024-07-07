import boto3
import json

brt = boto3.client(service_name='bedrock-runtime')
with open('MODEL_IDS.json','rt') as f:
    model_ids = json.load(f)['text2text']

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
                'content':[
                    {'text':'''あなたは論理問題に強い AI です。
文章から読み取れる事のみを回答し、答えられないことには必ず答えられません、と回答します。
<text> タグで文章を、<question> タグで質問を与えます。 AI は正しい回答だけをしてください。
思考など回答以外のを出力してはいけません。'''}
                ]
            },{
                'role':'assistant',
                'content': [
                    { 'text': '遵守します。' },
                ]
            },{
                'role':'user',
                'content': [
                    { 'text': '<text>吾輩は猫である。名前はまだない。</text><question>猫の名前は？</question>' },
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