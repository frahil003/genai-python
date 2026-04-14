# packages
from transformers import pipeline
from langchain_community.document_loaders import ArxivLoader

# model selection
task = "summarization"
model = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline(task=task, model=model)

# Data Preparation
query = "prompt engineering"
loader = ArxivLoader(query=query, load_max_docs=1)
docs = loader.load()

print(type(docs))

article_text = docs[0].page_content

# Run the summarization pipeline
result = summarizer(article_text[:2000], max_length=80, min_length=20, do_sample=False)

print(result[0]['summary_text'])

# Number of characters
print(len(result[0]['summary_text'].split(' ')))

