import re
from googletrans import Translator
from sentence_splitter import SentenceSplitter
from underthesea import sent_tokenize
import random

def clean_text(text):
    text = text.strip()
    lines = text.splitlines()

    clean_lines = [re.sub('\s+', ' ', line.strip())
                   for line in lines if line.strip()]

    replace_strings = ["\u003E\u003E", "Khóa Học", "Biên Phiên Dịch", "Tiếng Anh", "Biên - Phiên Dịch",
                       "Viết Email", "Giao Tiếp", "Ielts Essay", "Viết Ielts Essay",
                       "Ngữ Pháp Nâng Cao", "Biên Dịch", "Nghe - Hiểu", "Thương Mại", ]

    for idx, line in enumerate(clean_lines):
        for replace_string in replace_strings:
            line = line.replace(replace_string, "")
            line = line.replace(". .", ".")
        clean_lines[idx] = line

    return "\n".join(clean_lines)

def delete_sentences(text, start_word, end_word):
    sentences = text.split(".")
    new_sentences = []
    deleting = False

    for sentence in sentences:
        if start_word in sentence:
            deleting = True
            index = sentence.find(start_word)
            new_sentence = sentence[:index].strip()
            if new_sentence:
                new_sentences.append(new_sentence)
        elif end_word in sentence and deleting:
            deleting = False
            index = sentence.find(end_word) + len(end_word)
            new_sentence = sentence[index:].strip()
            if new_sentence:
                new_sentences.append(new_sentence)
        elif not deleting:
            new_sentences.append(sentence.strip())

    result = ". ".join(new_sentences)
    return result

def shuffle_document(text):
    shuf_text = text[:]
    return random.sample(shuf_text, len(text))


# Create a list to store the old and new positions of a suffle list
def position_changes(org_text, suff_text):
    pos = []
    # Compare positions of items in the shuffled list with the original list
    for idx, item_original in enumerate(org_text):
        new_pos = suff_text.index(item_original)
        pos.append(([idx], [new_pos]))
    return pos

def batchify(data, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size
    batches = [data[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    return batches

def detect_lang(text):
    translator = Translator(service_urls=[
      'translate.google.com.hk',
    ])
    max_len = 200
    chunk = text[0 : min(max_len, len(text))]
    lang = translator.detect(chunk).lang
    if lang.startswith('zh'):
        lang = 'zh'
    return lang

def split_sents(text, lang):
    if lang in LANG.SPLITTER:
        if lang == 'vi':
            sents = sent_tokenize(text)
        else:
            splitter = SentenceSplitter(language=lang)
            sents = splitter.split(text=text)
            sents = [sent.strip() for sent in sents]
        return sents
    else:
        raise Exception('The language {} is not suppored yet.'.format(LANG.ISO[lang]))
    
        
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
    
class LANG:
    SPLITTER = {
        'ca': 'Catalan',
        'zh': 'Chinese',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'en': 'English',
        'fi': 'Finnish',
        'fr': 'French',
        'de': 'German',
        'el': 'Greek',
        'hu': 'Hungarian',
        'is': 'Icelandic',
        'it': 'Italian',
        'lt': 'Lithuanian',
        'lv': 'Latvian',
        'no': 'Norwegian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'es': 'Spanish',
        'sv': 'Swedish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
    }
    ISO = {
		'aa': 'Afar',
		'ab': 'Abkhaz',
		'af': 'Afrikaans',
		'ak': 'Akan',
		'am': 'Amharic',
		'an': 'Aragonese',
		'ar': 'Arabic',
		'as': 'Assamese',
		'av': 'Avaric',
		'ay': 'Aymara',
		'az': 'Azerbaijani',
		'ba': 'Bashkir',
		'be': 'Belarusian',
		'bg': 'Bulgarian',
		'bh': 'Bihari',
		'bi': 'Bislama',
		'bm': 'Bambara',
		'bn': 'Bengali',
		'bo': 'Tibetan',
		'br': 'Breton',
		'bs': 'Bosnian',
		'ca': 'Catalan',
		'ce': 'Chechen',
		'ch': 'Chamorro',
		'co': 'Corsican',
		'cr': 'Cree',
		'cs': 'Czech',
		'cv': 'Chuvash',
		'cy': 'Welsh',
		'da': 'Danish',
		'de': 'German',
		'dv': 'Divehi',
		'dz': 'Dzongkha',
		'ee': 'Ewe',
		'el': 'Greek',
		'en': 'English',
		'es': 'Spanish',
		'et': 'Estonian',
		'eu': 'Basque',
		'fa': 'Persian',
		'ff': 'Fula',
		'fi': 'Finnish',
		'fj': 'Fijian',
		'fo': 'Faroese',
		'fr': 'French',
		'fy': 'Western Frisian',
		'ga': 'Irish',
		'gd': 'Scottish Gaelic',
		'gl': 'Galician',
		'gn': 'Guaraní',
		'gu': 'Gujarati',
		'gv': 'Manx',
		'ha': 'Hausa',
		'he': 'Hebrew',
		'hi': 'Hindi',
		'ho': 'Hiri Motu',
		'hr': 'Croatian',
		'ht': 'Haitian',
		'hu': 'Hungarian',
		'hy': 'Armenian',
		'hz': 'Herero',
		'id': 'Indonesian',
		'ig': 'Igbo',
		'ii': 'Nuosu',
		'ik': 'Inupiaq',
		'io': 'Ido',
		'is': 'Icelandic',
		'it': 'Italian',
		'iu': 'Inuktitut',
		'ja': 'Japanese',
		'jv': 'Javanese',
		'ka': 'Georgian',
		'kg': 'Kongo',
		'ki': 'Kikuyu',
		'kj': 'Kwanyama',
		'kk': 'Kazakh',
		'kl': 'Kalaallisut',
		'km': 'Khmer',
		'kn': 'Kannada',
		'ko': 'Korean',
		'kr': 'Kanuri',
		'ks': 'Kashmiri',
		'ku': 'Kurdish',
		'kv': 'Komi',
		'kw': 'Cornish',
		'ky': 'Kyrgyz',
		'lb': 'Luxembourgish',
		'lg': 'Ganda',
		'li': 'Limburgish',
		'ln': 'Lingala',
		'lo': 'Lao',
		'lt': 'Lithuanian',
		'lu': 'Luba-Katanga',
		'lv': 'Latvian',
		'mg': 'Malagasy',
		'mh': 'Marshallese',
		'mi': 'Māori',
		'mk': 'Macedonian',
		'ml': 'Malayalam',
		'mn': 'Mongolian',
		'mr': 'Marathi',
		'ms': 'Malay',
		'mt': 'Maltese',
		'my': 'Burmese',
		'na': 'Nauru',
		'nb': 'Norwegian Bokmål',
		'nd': 'North Ndebele',
		'ne': 'Nepali',
		'ng': 'Ndonga',
		'nl': 'Dutch',
		'nn': 'Norwegian Nynorsk',
		'no': 'Norwegian',
		'nr': 'South Ndebele',
		'nv': 'Navajo',
		'ny': 'Chichewa',
		'oc': 'Occitan',
		'oj': 'Ojibwe',
		'om': 'Oromo',
		'or': 'Oriya',
		'os': 'Ossetian',
		'pa': 'Panjabi',
		'pl': 'Polish',
		'ps': 'Pashto',
		'pt': 'Portuguese',
		'qu': 'Quechua',
		'rm': 'Romansh',
		'rn': 'Kirundi',
		'ro': 'Romanian',
		'ru': 'Russian',
		'rw': 'Kinyarwanda',
		'sa': 'Sanskrit',
		'sc': 'Sardinian',
		'sd': 'Sindhi',
		'se': 'Northern Sami',
		'sg': 'Sango',
		'si': 'Sinhala',
		'sk': 'Slovak',
		'sl': 'Slovenian',
		'sm': 'Samoan',
		'sn': 'Shona',
		'so': 'Somali',
		'sq': 'Albanian',
		'sr': 'Serbian',
		'ss': 'Swati',
		'st': 'Southern Sotho',
		'su': 'Sundanese',
		'sv': 'Swedish',
		'sw': 'Swahili',
		'ta': 'Tamil',
		'te': 'Telugu',
		'tg': 'Tajik',
		'th': 'Thai',
		'ti': 'Tigrinya',
		'tk': 'Turkmen',
		'tl': 'Tagalog',
		'tn': 'Tswana',
		'to': 'Tonga',
		'tr': 'Turkish',
		'ts': 'Tsonga',
		'tt': 'Tatar',
		'tw': 'Twi',
		'ty': 'Tahitian',
		'ug': 'Uighur',
		'uk': 'Ukrainian',
		'ur': 'Urdu',
		'uz': 'Uzbek',
		've': 'Venda',
		'vi': 'Vietnamese',
		'wa': 'Walloon',
		'wo': 'Wolof',
		'xh': 'Xhosa',
		'yi': 'Yiddish',
		'yo': 'Yoruba',
		'za': 'Zhuang',
		'zh': 'Chinese',
		'zu': 'Zulu',
    }
