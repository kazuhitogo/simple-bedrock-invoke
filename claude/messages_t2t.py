# docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

import boto3
import json

br = boto3.client(service_name='bedrock-runtime')

ANTHROPIC_VERSION = 'bedrock-2023-05-31'

AVAILABLE_MODEL_IDS = [
    'anthropic.claude-instant-v1', # v1
    'anthropic.claude-v2:1','anthropic.claude-v2', # v2
    'anthropic.claude-3-sonnet-20240229-v1:0', # v3
]
model_id = AVAILABLE_MODEL_IDS[-1] # claude-3-sonnet


system_prompt = '以下はユーザーと AI のやりとりです。AI はユーザーから <knowledge></knowledge> で囲って与えられた情報だけを使ってユーザーからの質問に<output>タグで括って端的に答え、余計な修飾は行いません。'

messages =  [
    {'role': 'user', 'content': '''<knowledge>
Amazon Titan とは
Amazon Bedrock限定のAmazon Titanファミリーモデルには、ビジネス全体でAIと機械学習の革新に取り組んできたAmazonの25年の経験が組み込まれています。Amazon Titan 基盤モデル (FM) は、フルマネージド API を通じて、高性能な画像、マルチモーダル、テキストモデルの選択肢を幅広くお客様に提供します。Amazon Titan モデルは AWS によって作成され、大規模なデータセットで事前にトレーニングされているため、さまざまなユースケースをサポートすると同時に、AI の責任ある使用をサポートするように構築された強力な汎用モデルとなっています。そのまま使用することも、独自のデータを使用して個人的にカスタマイズすることもできます。
</knowledge>
'''},
    {'role': 'assistant', 'content': '覚えました。'},
    {'role': 'user', 'content': 'Amazon Titan には Amazon の何年の経験が組み込まれていますか？'},
    {'role': 'assistant', 'content': '<output>'},
]

body=json.dumps({
    'anthropic_version': ANTHROPIC_VERSION,
    'max_tokens': 4096,
    'system': system_prompt,
    'messages': messages,
    'temperature': 0,
    'top_p': 0,
    'top_k': 0,
    'stop_sequences': ['</output>']
})

response = br.invoke_model(body=body, modelId=model_id)
response_body = json.loads(response.get('body').read())
print(response_body['content'][0]['text'])

