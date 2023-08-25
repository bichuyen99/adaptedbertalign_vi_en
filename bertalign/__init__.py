"""
Bertalign initialization
"""

from bertalign.encoder import Encoder
from bertalign.prepdata import *
from bertalign.utils import *
from bertalign.eval import *
from bertalign.aligner import Bertalign

source = 'data/songngu.txt'
word_vi = 'Theo:'
word_en = 'Source:'
model_fasttext = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")

model_name = "LaBSE"
model = Encoder(model_name)

# Document alignment
doc_en, doc_vi = preprocessing_data(source, word_vi, word_en, model_fasttext)
batch_size = 32
batches_en = batchify(doc_en[:320], batch_size)
batches_vi = batchify(doc_vi[:320], batch_size)
iter = 0

def align_docs(batches_en, batches_vi):
	for batch_idx, (batch_en, batch_vi) in enumerate(zip(batches_en, batches_vi)):
	    iter = batch_idx + 1
	    print(f"Batch {iter}:")
	    batch_vi_new = shuffle_document(batch_vi)
	    pos = position_changes(batch_vi, batch_vi_new)
	    en = create_document(batch_en, add_line_break=True)
	    vi = create_document(batch_vi_new, add_line_break=True)
	    aligner = Bertalign(en, vi, is_split=True)
	    aligner.align_sents()
	    test_alignments = []
	    test_alignments.append(aligner.result)
	    scores = score_multiple(pos,test_alignments[0])
	    log_final_scores(scores)
	    # Calculate the sum of each score type
	    total_recall_strict += scores['recall_strict']
	    total_recall_lax += scores['recall_lax']
	    total_precision_strict += scores['precision_strict']
	    total_precision_lax += scores['precision_lax']
	    total_f1_strict += scores['f1_strict']
	    total_f1_lax += scores['f1_lax']
	avg = dict(recall_strict=total_recall_strict/iter,
	           recall_lax=total_recall_lax/iter,
	           precision_strict=total_precision_strict/iter,
	           precision_lax=total_precision_lax/iter,
	           f1_strict=total_f1_strict/iter,
	           f1_lax=total_f1_lax/iter)

	print(' ---------------------------------')
	print('Average scores:')
	log_final_scores(avg)

#Sentence alignment
def align_sents(number, doc_en, doc_vi):
	for i in range(number):
	  print("\nDocument number ", i+1)
	  aligner = Bertalign(doc_en[i], doc_vi[i], is_split=False)
	  aligner.align_sents()
	  aligner.print_sents()