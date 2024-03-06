# docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html

import boto3
import json
br = boto3.client(service_name='bedrock-runtime')

AVAILABLE_MODEL_IDS = [
    'anthropic.claude-instant-v1', # v1
    'anthropic.claude-v2', 'anthropic.claude-v2:1', # v2
]

modelId = AVAILABLE_MODEL_IDS[-1] # v2:1
accept = 'application/json'
contentType = 'application/json'

prompt = '''以下はユーザーと AI のやりとりです。AI はユーザーから <knowledge></knowledge> で囲って与えられた情報だけを使ってユーザーからの質問に<output>タグで括って端的に答え、余計な修飾は行いません。

Human: <knowledge>
Amazon Titan とは
Amazon Bedrock限定のAmazon Titanファミリーモデルには、ビジネス全体でAIと機械学習の革新に取り組んできたAmazonの25年の経験が組み込まれています。Amazon Titan 基盤モデル (FM) は、フルマネージド API を通じて、高性能な画像、マルチモーダル、テキストモデルの選択肢を幅広くお客様に提供します。Amazon Titan モデルは AWS によって作成され、大規模なデータセットで事前にトレーニングされているため、さまざまなユースケースをサポートすると同時に、AI の責任ある使用をサポートするように構築された強力な汎用モデルとなっています。そのまま使用することも、独自のデータを使用して個人的にカスタマイズすることもできます。
</knowledge>

Assistant: 覚えました。

Human: Amazon Titan には Amazon の何年の経験が組み込まれていますか？

Assistant: <output>'''

body = json.dumps({
    'prompt': prompt,
    'max_tokens_to_sample': 4096,
    'temperature': 0,
    'top_p': 0,
    'top_k': 0,
    'stop_sequences': ['</output>'],
})

response = br.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)

stream = response.get('body')

if stream:
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            print(json.loads(chunk.get('bytes').decode())['completion'],end='')
metrics = json.loads(chunk.get('bytes').decode())['amazon-bedrock-invocationMetrics']
print(metrics)