import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

def get_stopwords():
    return set(stopwords.words('english'))

def get_pos(words: list[str], tagset='universal'):
    word_n_pos_list = nltk.pos_tag(words, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list

def pos_filter(ori_pos, new_pos_list):
    return [ori_pos == new_pos or {ori_pos, new_pos} <= {'NOUN', 'VERB'} for new_pos in new_pos_list]