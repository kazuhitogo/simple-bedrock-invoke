# docs:
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html

import json
import boto3

br = boto3.client(service_name='bedrock-runtime',region_name='us-west-2')

AVAILABLE_MODEL_IDS = [
    'mistral.mistral-7b-instruct-v0:2', # 7b
    'mistral.mixtral-8x7b-instruct-v0:1', # 8x7b
    'mistral.mistral-large-2402-v1:0' # large
]

modelId = AVAILABLE_MODEL_IDS[0] # v2:1

# prompt
phrase = 'Amazon Bedrockは、高いパフォーマンスを提供する基底モデル（FM）を提供するLeading AI社の1つから複数のAmazon、AI21 Labs、Anthropic、Cohere、Meta、Mistral AI、Stability AIによって管理される全管理サービスです。 これには、セキュリティ、プライバシー、責任のあるAIアプリケーションを構築するために必要な広範な機能が含まれています。 Amazon Bedrockを使用することで、使用ケースに適したFMを簡単に実験・評価でき、自分のデータでプライベートにカスタマイズでき、企業システムやデータソースを使用してタスクを実行するエージェントを構築できます。Amazon Bedrockはサーバレスであるため、インフラストラクチャを管理する必要はありません。あなたが熟知しているAWSサービスを使用して、アプリケーションに生成AI能力を安全に統合・デプロイできます。'
from_lang = '日本語'
to_lang = '英語'
prompt = f'''<s>[INST] あなたはチャットでユーザを支援するAIアシスタントです。日本語で会話をしてください。 [/INST]
コンテキストを理解しました。</s>
[INST] これから{from_lang}の文章を与えるので、{to_lang}に訳してください。ただし、出力は <output> 翻訳結果 </output> としてください。 [/INST]
わかりました。</s> 
[INST] {phrase} [/INST]
<output>'''

body = json.dumps({
    'prompt': prompt,
    'max_tokens': 4096,
    'temperature': 0.7,
    'top_p': 0.7,
    'top_k': 50,
    'stop' : ['</output>']
})

response = br.invoke_model_with_response_stream(body=body, modelId=modelId)
for event in response.get("body"):
    chunk = json.loads(event["chunk"]["bytes"])['outputs'][0]
    print(chunk['text'],end='')
print(f'\nstop_reason: {chunk["stop_reason"]}')
