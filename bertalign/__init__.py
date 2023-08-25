"""
Bertalign initialization
"""

from bertalign.prepdata import *
from bertalign.utils import *
from bertalign.aligner import *

source = 'data/songngu.txt'
word_vi = 'Theo:'
word_en = 'Source:'
model_fasttext = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")


doc_en, doc_vi = preprocessing_data(source, word_vi, word_en, model_fasttext)

# Detect if the English document contains Vietnamese sentences, then delete these sentences.
model_ft = fasttext.load_model(model_fasttext)
for i in range(len(doc_en)):
	sentences = nltk.sent_tokenize(doc_en[i])
	result = model_ft.predict(sentences[0])
	if result[0][0] == '__label__eng_Latn' and result[1][0] >= 0.99:
		pass
	else:
		doc_en[i] = ' '.join(sentences[1:])

batch_size = 32
batches_en = batchify(doc_en[:320], batch_size)
batches_vi = batchify(doc_vi[:320], batch_size)

# Document alignment
align_documents(batches_en, batches_vi)

# Sentence alignment
number = 10
align_sentences(number, doc_en, doc_vi)
