import requests
import os

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = os.getenv('1ba68d2ffcbdad1ac7dbc992797cfa0200a9031ab7c886e6701674892ba4acbf')

res = requests.post(endpoint, json={
    "model": 'togethercomputer/RedPajama-INCITE-7B-Base',
    "prompt": """\
      Label the sentences as either "positive", "negative", "mixed", or "neutral":

      Sentence: I can say that there isn't anything I would change.
      Label: positive

      Sentence: I'm not sure about this.
      Label: neutral

      Sentence: I liked some parts but I didn't like other parts.
      Label: mixed

      Sentence: I think the background image could have been better.
      Label: negative

      Sentence: I really like it.
      Label:""",
    "top_p": 1,
    "top_k": 40,
    "temperature": 0.8,
    "max_tokens": 1,
    "repetition_penalty": 1,
}, headers={
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "User-Agent": "<YOUR_APP_NAME>"
})
print(res.json()['output']['choices'][0]['text']) # ' positive'