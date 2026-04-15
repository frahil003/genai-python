from transformers import pipeline
import pprint

unmasker = pipeline(task='fill-mask', model='bert-base-uncased')
result = unmasker("I love expensive clothes and I am a [MASK] model.")

for item in result:
    print(f"{item['token_str']}: {item['score']:.4f}: {item['sequence']}")


