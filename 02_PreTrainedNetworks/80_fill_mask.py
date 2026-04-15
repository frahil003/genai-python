from transformers import pipeline
from pprint import pprint

unmasker = pipeline(task='fill-mask', model='bert-base-uncased')
result = unmasker("I am a [MASK] model")

for i, item in enumerate(result, start=1):
    print(
        f"{i}. token={item['token_str']!r}, "
        f"score={item['score']:.4f}, "
        f"text={item['sequence']}"
    )

print('#'*50)

pprint(result)



