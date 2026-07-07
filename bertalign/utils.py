import re
import asyncio
from googletrans import Translator
from sentence_splitter import SentenceSplitter
from bertalign.languages import LANG

def clean_text(text):
    clean_text = []
    text = text.strip()
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r'\s+', ' ', line)
            clean_text.append(line)
    return "\n".join(clean_text)
    
def detect_lang(text):
    if not text or not text.strip():
        return "en"
    
    async def _async_detect():
        translator = Translator(service_urls=[
            'translate.google.com.hk',
        ])
        max_len = 200
        chunk = text[:min(max_len, len(text))]
        lang = (await translator.detect(chunk)).lang
        if lang.startswith('zh'):
            lang = 'zh'
        return lang

    return asyncio.run(_async_detect())

def is_language_supported(lang):
    return lang in LANG.SPLITTER

def check_language(lang):
    if not is_language_supported(lang):
        raise Exception('The language {} is not suppored yet.'.format(LANG.ISO[lang]))

def split_sents(text, lang):
    check_language(lang)
    if lang == 'zh':
        sents = _split_zh(text)
    else:
        splitter = SentenceSplitter(language=lang)
        sents = splitter.split(text=text) 
        sents = [sent.strip() for sent in sents]
    return sents
    
def _split_zh(text, limit=1000):
        sent_list = []
        text = re.sub('(?P<quotation_mark>([。？！](?![”’"\'）])))', r'\g<quotation_mark>\n', text)
        text = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'）])', r'\g<quotation_mark>\n', text)

        sent_list_ori = text.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)

        return sent_list
        
def yield_overlaps(lines, num_overlaps):
    lines = [_preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in _layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2

def _layer(lines, num_overlaps, comb=' '):
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii:ii + num_overlaps]))
    return out
    
def _preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line

