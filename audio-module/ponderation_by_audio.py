import re
import string
import os
from gensim import corpora, models, similarities
from imdb import IMDb
from audio_utils import load_stoplists, remove_stopwords, remove_low_freq
from imdb_utils import get_plots

####################################################################################################

backend_folder = os.getcwd()
path_base = os.path.dirname(__file__)
path_stoplist = os.path.join(path_base, 'stopwords.txt')
path_text_audio = os.path.join(path_base, 'text_audio.txt')

####################################################################################################


# returns a dict (movie_id, similarity)
def movie_similarity_by_audio(movies_ids, plots, verbose=False):
    if verbose:
        print("[PON-AUD] Iniciando ponderacion por audio...")
    
    similarity_by_movie = {}

    # get text listened from audio
    with open(path_text_audio, 'r') as reader:
        text_audio = reader.read()

    text_audio = preprocess_text(text_audio)

    texts = []
    docs = plots

    # remove stopwords and words with frequency = 1 from docs
    stoplist = load_stoplists(path_stoplist)
    for doc in docs:
        preprocessed_text = remove_stopwords(doc, stoplist)
        preprocessed_text = remove_low_freq(preprocessed_text, 1)
        flat_list = [item for sublist in preprocessed_text for item in sublist]
        texts.append(flat_list)
    
    # doc2bow counts the number of occurences of each distinct word
    if verbose:
        print("[PON-AUD] Creando diccionario...")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=movies_ids.__len__())
    if verbose:
        print("[PON-AUD] Diccionario creado.")

    vec_bow = dictionary.doc2bow(text_audio.lower().split())

    if verbose:
        print("[PON-AUD] Ponderando peliculas...")
    # convert the query to LSI space
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])

    # perform a similarity query against the corpus
    sims = index[vec_lsi]
    
    for i in range(0, movies_ids.__len__()):
        similarity_by_movie[movies_ids[i]] = str(sims[i])  # para que despues se pueda serializar: Object of type float32 is not JSON serializable
    
    if verbose:
        print("[PON-AUD] Ponderacion completa.")

    #print(lsi.print_topics())
    return similarity_by_movie


def preprocess_text(text):
    text = re.sub("\'\w+ ", " ", text)
    return text.translate(text.maketrans('', '', string.punctuation + string.digits))

####################################################################################################