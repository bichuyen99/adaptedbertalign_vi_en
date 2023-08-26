import numpy as np

from bertalign.corelib import *
from bertalign.utils import *
from bertalign.eval import *
from bertalign.encoder import Encoder

model_name = "LaBSE"
model = Encoder(model_name)

class Bertalign:
    def __init__(self,
                 src,
                 tgt,
                 max_align=5,
                 top_k=1,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 is_split=False,
               ):

        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty

        src_lang = detect_lang(src)
        tgt_lang = detect_lang(tgt)

        if is_split:
             src_sents = src.splitlines()
             tgt_sents = tgt.splitlines()
        else:
             src_sents = split_sents(src, src_lang)
             tgt_sents = split_sents(tgt, tgt_lang)

        src_num = len(src_sents)
        tgt_num = len(tgt_sents)

        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]

        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        print("Embedding source and target text using {} ...".format(model.model_name))
        src_vecs, src_lens = model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs
        self.is_split = is_split

    def align_sents(self):

        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment_1, first_alignment_2 = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)

        if self.is_split:
            print("Finished! Successfully aligning {} {} documents to {} {} documents".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
            self.result = first_alignment_1
        else:
            print("Performing second-step alignment ...")
            second_alignment_types = get_alignment_types(self.max_align)
            second_w, second_path = find_second_search_path(first_alignment_2, self.win, self.src_num, self.tgt_num)
            second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                                second_w, second_path, second_alignment_types,
                                                self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
            second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)

            print("Finished! Successfully aligning {} {} sentences to {} {} sentences".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
            self.result = second_alignment

    def print_sents(self):
         with open('output.txt', 'a') as f:
            for bead in (self.result):
                src_line = self._get_line(bead[0], self.src_sents)
                tgt_line = self._get_line(bead[1], self.tgt_sents)
                f.write(src_line + "\n" + tgt_line + "\n")
            f.write("\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line


# Document alignment
def align_documents(batches_en, batches_vi):
    iter = 0
    # Initialize variables to store the sum of each score type
    total_recall_strict = 0
    total_recall_lax = 0
    total_precision_strict = 0
    total_precision_lax = 0
    total_f1_strict = 0
    total_f1_lax = 0
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

# Sentence alignment
def align_sentences(number, doc_en, doc_vi):
    for i in range(number):
      print("\nDocument number ", i+1)
      aligner = Bertalign(doc_en[i], doc_vi[i], is_split=False)
      aligner.align_sents()
      aligner.print_sents()
