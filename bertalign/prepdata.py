from bertalign.utils import *
import fasttext
from huggingface_hub import hf_hub_download

def preprocessing_data(source, word_vi, word_en, model_fasttext):
	# Load the data
	data = []
	with open(source, 'r') as file:
		data = [eval(line) for line in file]

	# Split the data into English and Vietnamese sentences
	sents_en = [clean_text(delete_sentences(entry["article_en"],word_en,'url')) for entry in data]
	sents_vi = [clean_text(delete_sentences(entry["article_vn"],word_vi,'url')) for entry in data]

	# Create lists of English and Vietnamese documents without empty one
	empty_sents = []
	for i in range(len(sents_en)):
	  if sents_en[i] == '' or sents_vi[i] == '':
	      empty_sents.append(i)
	doc_en = [sents_en[i] for i in range(len(sents_en)) if i not in empty_sents]
	doc_vi = [sents_vi[i] for i in range(len(sents_vi)) if i not in empty_sents]

	return doc_en, doc_vi
