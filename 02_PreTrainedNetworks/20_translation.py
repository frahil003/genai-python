from transformers import pipeline

# model selection
task = "translation"
# model for english -> japanese
# model = "Mitsua/elan-mt-bt-en-ja"

# model for english -> german
model = "Helsinki-NLP/opus-mt-en-de"

translator = pipeline(task=task, model=model)

text = "Be the change you wish to see in the world"
result = translator(text)
print(result[0]['translation_text'])

