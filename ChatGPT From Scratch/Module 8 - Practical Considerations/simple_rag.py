

from datasets import load_dataset
import torch
import numpy as np
from openai import OpenAI
import os
# I had iomport problems
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1"); os.environ.setdefault("DISABLE_TORCHVISION", "1")
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-30m-english", trust_remote_code=True)
model = AutoModel.from_pretrained("ibm-granite/granite-embedding-30m-english", trust_remote_code=True)

HFTOKEN = "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"  # Replace with your token or use environment variable

class SimpleRAGNews():

	def __init__(self):

		# load the dataset "permutans/fineweb-bbc-news" in streaming mode,
		# using the subset: "CC-MAIN-2013-20"
		# and the split: "train"
		self.dataset = load_dataset(
			"permutans/fineweb-bbc-news",
			name="CC-MAIN-2013-20",
			split="train",
			streaming=True
		)

		# load the model "ibm-granite/granite-embedding-30m-english"
		# and corresponding tokenizer using AutoModel and AutoTokenizer
		# as per the instructions
		self.tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-embedding-30m-english')
		self.model = AutoModel.from_pretrained('ibm-granite/granite-embedding-30m-english')
		self.model.eval()  # Set to evaluation mode

		self.setup_db()

		# finally, create a client to use huggingface inference
		self.client = OpenAI(
		    base_url="https://router.huggingface.co/v1",
		    api_key=HFTOKEN,
		)


	def setup_db(self):
		# take the first 100 samples from the bbc-news dataset
		# pare down the columns and only keep the "text" column
		# then convert to a list: list(ds) for easy retrieval later
		ds = self.dataset.take(100)
		ds = ds.remove_columns([col for col in ds.column_names if col != 'text'])
		self.articles = list(ds)

		# for each entry in the dataset, call self.embed() on the text
		# save these vectors for later use
		self.embeddings = []
		for article in self.articles:
			embedding = self.embed(article['text'])
			self.embeddings.append(embedding)
		
		# Stack embeddings into a tensor for efficient similarity computation
		self.embeddings = torch.cat(self.embeddings, dim=0)


	def embed(self, text):
		# given a passed string, tokenize the text using the ibm-granitetokenizer

		# hint: make sure you pass the following to the tokenizer:
		# padding=True, truncation=True, return_tensors='pt'
		tokens = self.tokenizer(
			text,
			padding=True,
			truncation=True,
			return_tensors='pt'
		)

		# then pass through the embedding model as shown on the ibm-granite page:
		# embedding = model(**tokens)[0][:, 0]
		# embedding = torch.nn.functional.normalize(embedding, dim=1)
		with torch.no_grad():
			embedding = self.model(**tokens)[0][:, 0]
			embedding = torch.nn.functional.normalize(embedding, dim=1)
		
		return embedding


	def get_most_relevant_news_article_text(self, user_query):
		# given a user query, this method should:
		# call self.embed(user_query)
		query_embedding = self.embed(user_query)
		
		# compare the embedding against all stored embeddings using
		# torch.nn.functional.cosine_similarity
		similarities = torch.nn.functional.cosine_similarity(
			query_embedding,
			self.embeddings,
			dim=1
		)
		
		# use the top match to return the text of most relevant article
		top_idx = torch.argmax(similarities).item()
		return self.articles[top_idx]['text']


	def summarize_article(self, article):
		# Use the HF Inference API to ask "openai/gpt-oss-20b" to
		# summarize an article. 

		# This should then return the model's final response text.
		completion = self.client.chat.completions.create(
			model="openai/gpt-oss-20b",
			messages=[
				{
					"role": "system",
					"content": "You are a helpful assistant that summarizes news articles concisely."
				},
				{
					"role": "user",
					"content": f"Please summarize the following news article:\n\n{article}"
				}
			],
			max_tokens=150
		)
		
		return completion.choices[0].message.content


	def summary_for_query(self, query):
		# get the most relevant article
		article_text = self.get_most_relevant_news_article_text(query)
		
		# get a summary of the article
		summary = self.summarize_article(article_text)
		
		# return to user
		return summary



if __name__ == "__main__":

	rag = SimpleRAGNews()
	query = "california wildfires"
	news_blurb_for_user = rag.summary_for_query(query)

	print("An AI-generated summary of the most relevant article:")
	print(news_blurb_for_user)

	# EXPECTED OUTPUT:
	# An AI-generated summary of the most relevant article:
	# **Summary**

	# California’s current wildfire crisis shows a markedly different federal and state response compared with Hurricane Katrina.  
	# - **Federal involvement**: FEMA, Homeland Security, the

"""
ACTUAL OUPUT:
 C:\Users\Putna\OneDrive - Johns Hopkins\Documents\Johns Hopkins\ChatGPT From Scratch\Module 8 - Practical Considerations>$env:TRANSFORMERS_NO_TORCHVISION="1"; $env:DISABLE_TORCHVISION="1"; $env:PYTHONUNBUFFERED="1"; & "C:\Users\Putna\.conda\envs\en605645\python.exe" "C:\Users\Putna\OneDrive - Johns Hopkins\Documents\Johns Hopkins\ChatGPT From Scratch\Module 8 - Practical Considerations\simple_rag.py"
The filename, directory name, or volume label syntax is incorrect.
An AI-generated summary of the most relevant article:
**California Wildfires vs. Hurricane Katrina: Key Takeaways**

- **Federal Response Improvement**
  – The U.S. government has avoided the “chaotic, public‑facing” mess that marred Katrina.
  – Homeland Security’s Michael Chertoff and FEMA’s David Paulison are on site, praising a “very good team effort” among local, state, and
"""