import numpy as np
import logging

from bertalign.encoder import Encoder
from bertalign.corelib import *
from bertalign.utils import *
from lingua import LanguageDetector, Language, IsoCode639_1

class Bertalign:
    def __init__(self,
                 model_encoder: Encoder,
                 src,
                 tgt,
                 language_detector: LanguageDetector = None,
                 src_lang = None,
                 tgt_lang = None,
                 max_align = 5,
                 top_k = 3,
                 win = 5,
                 skip = -0.1,
                 margin = True,
                 len_penalty = True,
                 is_split = False,
                 progress_callback = None
               ):
        
        if not language_detector and (not src_lang or not tgt_lang):
            raise Exception('Can\'t detect languages without a language detector.')

        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        self.logger = logging.getLogger(__name__)
        self.progress_callback = progress_callback
        self.language_detector = language_detector

        src = clean_text(src)
        tgt = clean_text(tgt)

        src_lang = Language.from_iso_code_639_1(IsoCode639_1.from_str(src_lang)) if src_lang else self._detect_language(src)
        tgt_lang = Language.from_iso_code_639_1(IsoCode639_1.from_str(tgt_lang)) if tgt_lang else self._detect_language(tgt)
        
        if is_split:
            src_sents = src.splitlines()
            tgt_sents = tgt.splitlines()
        else:
            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)
 
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        self.logger.info(f"Source language: {src_lang.name}, Number of sentences: {src_num}")
        self.logger.info(f"Target language: {tgt_lang.name}, Number of sentences: {tgt_num}")

        src_vecs, src_lens = model_encoder.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = model_encoder.transform(tgt_sents, max_align - 1)

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
    
    def _detect_language(self, text) -> Language:
        self.logger.info(f"Detecting language: \"{text[:30]}...\"")
        lang = self.language_detector.detect_languages_in_parallel_of([text])
        if not lang:
            self.logger.info(f"Language not detected: \"{text[:30]}...\"")
            return Language.ENGLISH
        self.logger.info(f"Language {lang[0].name} detected: \"{text[:30]}...\"")
        return lang[0]

    def _set_progress(self, step: int, total: int):
        if self.progress_callback:
            self.progress_callback('bertalign', "Bertalign", step, total)

    def align_sents(self):
        self._set_progress(0, 2)
        self.logger.info("Performing first-step alignment ...")

        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        self._set_progress(1, 2)
        self.logger.info("Performing second-step alignment ...")

        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        self.result = second_alignment

        self._set_progress(2, 2)
        self.logger.info(f"Finished! Successfully aligned {self.src_num} {self.src_lang} sentences to {self.tgt_num} {self.tgt_lang} sentences\n")
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line
