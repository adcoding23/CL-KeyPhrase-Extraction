import os
import time
import sys
import re
from subprocess import call
import numpy as np

from nltk import TweetTokenizer, bigrams, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.parse.corenlp import CoreNLPParser as StanfordTokenizer 

#Reading files
from os import listdir
from os.path import isfile, join
    
FASTTEXT_EXEC_PATH = os.path.abspath("./fasttext")
BASE_SNLP_PATH = "http://localhost:9000/stanford-postagger-2018-10-16"
SNLP_TAGGER_JAR = os.path.join(BASE_SNLP_PATH, "stanford-postagger.jar")

MODEL_WIKI_UNIGRAMS = os.path.abspath("./wiki_unigrams.bin")
MODEL_WIKI_BIGRAMS = os.path.abspath("./wiki_bigrams.bin")
MODEL_TWITTER_UNIGRAMS = os.path.abspath('./twitter_unigrams.bin')
MODEL_TWITTER_BIGRAMS = os.path.abspath('./twitter_bigrams.bin')

def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    #print(sentence)
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    #print(sentence)
    if to_lower:
        sentence = sentence.lower()
        #print(sentence)
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    #print(sentence)
    sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    #print(sentence)
    filter(lambda word: ' ' not in word, sentence)
    #print(sentence)
    return sentence

def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]

def get_embeddings_for_preprocessed_sentences(sentences, model_path, fasttext_exec_path):
    """Arguments:
        - sentences: a list of preprocessed sentences
        - model_path: a path to the sent2vec .bin model
        - fasttext_exec_path: a path to the fasttext executable
    """
    timestamp = str(time.time())
    test_path = os.path.abspath('./'+timestamp+'_fasttext.test.txt')
    embeddings_path = os.path.abspath('./'+timestamp+'_fasttext.embeddings.txt')
    dump_text_to_disk(test_path, sentences)

    call(fasttext_exec_path+
          ' print-sentence-vectors '+
          model_path + ' < '+
          test_path + ' > ' +
          embeddings_path, shell=True)
    embeddings = read_embeddings(embeddings_path)
    os.remove(test_path)
    os.remove(embeddings_path)
    assert(len(sentences) == len(embeddings))
    return np.array(embeddings)

def read_embeddings(embeddings_path):
    """Arguments:
        - embeddings_path: path to the embeddings
    """
    with open(embeddings_path, 'r') as in_stream:
        embeddings = []
        for line in in_stream:
            line = '['+line.replace(' ',',')+']'
            embeddings.append(eval(line))
        return embeddings
    return []

def dump_text_to_disk(file_path, X, Y=None):
    """Arguments:
        - file_path: where to dump the data
        - X: list of sentences to dump
        - Y: labels, if any
    """
    with open(file_path, 'w') as out_stream:
        if Y is not None:
            for x, y in zip(X, Y):
                out_stream.write('__label__'+str(y)+' '+x+' \n')
        else:
            for x in X:
                out_stream.write(x+' \n')

def get_sentence_embeddings(sentences, phrase = True, ngram='bigrams', model='concat_wiki_twitter'):
    """ Returns a numpy matrix of embeddings for one of the published models. It
    handles tokenization and can be given raw sentences.
    Arguments:
        - ngram: 'unigrams' or 'bigrams'
        - model: 'wiki', 'twitter', or 'concat_wiki_twitter'
        - sentences: a list of raw sentences ['Once upon a time', 'This is another sentence.', ...]
    """
    wiki_embeddings = None
    twitter_embbedings = None
    tokenized_sentences_NLTK_tweets = None
    tokenized_sentences_SNLP = None
    if model == "wiki" or model == 'concat_wiki_twitter':
        tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
        #print("sentences",sentences) 
        #print(tknzr)
        s = ' <delimiter> '.join(sentences) #just a trick to make things faster
        #print("S",s)
        tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
        tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ')
        assert(len(tokenized_sentences_SNLP) == len(sentences))

        if phrase:
            temp = extract_keyphrase_candidates(str(tokenized_sentences_SNLP))
            #print("Temp = ", temp)
            temp1 = list([' '.join(x) for x in temp if type(x) == list])
            temp2 = list([x for x in temp if type(x) == str])
            tokenized_sentences_SNLP = [*temp1, *temp2]
            
        #print("Senetences = ", sentences, "\n")
        if ngram == 'unigrams':
            wiki_embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                     MODEL_WIKI_UNIGRAMS, FASTTEXT_EXEC_PATH)
        else:
            wiki_embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                     MODEL_WIKI_BIGRAMS, FASTTEXT_EXEC_PATH)
    if model == "twitter" or model == 'concat_wiki_twitter':
        tknzr = TweetTokenizer()
        tokenized_sentences_NLTK_tweets = tokenize_sentences(tknzr, sentences)
        if ngram == 'unigrams':
            twitter_embbedings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_NLTK_tweets, \
                                     MODEL_TWITTER_UNIGRAMS, FASTTEXT_EXEC_PATH)
        else:
            twitter_embbedings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_NLTK_tweets, \
                                     MODEL_TWITTER_BIGRAMS, FASTTEXT_EXEC_PATH)
    if model == "twitter":
        return twitter_embbedings
    elif model == "wiki":
        if phrase:
            return tokenized_sentences_SNLP, wiki_embeddings
        else:
            return wiki_embeddings
    elif model == "concat_wiki_twitter":
        return np.concatenate((wiki_embeddings, twitter_embbedings), axis=1)
    sys.exit(-1)


    
## POS Tagging
from nltk.parse import CoreNLPParser
#from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
parser = CoreNLPParser(url='http://localhost:9000')
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
   
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
   
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
   
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
   
    # Return a list of words
    return(text)

def extract_keyphrase_candidates(text):

    tagger = list(pos_tagger.tag(text.split()))
    keyphrase_candidates =[]
    phrase = []
    phrase_noun = []
    is_adj_candidate = False
    is_multinoun_candidate = False
    i =0
    while i<len(tagger):
        # adjectives + nouns
        if tagger[i][1] == 'JJ':
            is_adj_candidate = True
            phrase.append(tagger[i][0])
        if tagger[i][1] == 'NN' and is_adj_candidate:
            phrase.append(tagger[i][0])
        if tagger[i][1] == 'NN' and not is_adj_candidate:
            keyphrase_candidates.append(tagger[i][0])
        elif len(phrase) >= 2:
            keyphrase_candidates.append(phrase)
            is_adj_candidate = False
            phrase = []
       
        # multiple nouns
        if tagger[i][1] == 'NN':
            phrase_noun.append(tagger[i][0])
            is_multinoun_candidate = True
        elif len(phrase_noun) >= 2:
            keyphrase_candidates.append(phrase_noun)

            is_multinoun_candidate = False
            phrase_noun = []
        else:
            is_multinoun_candidate = False
            phrase_noun = []

        i = i+1
    #print(keyphrase_candidates)
    return keyphrase_candidates

def mmr1(phrases, phrase_embeddings, document_embeddings, l = 0.8, topk = 5):

        doc_similarity = cosine_similarity(phrase_embeddings, document_embeddings.reshape(1, -1))
        phrase_similarity_matrix = cosine_similarity(phrase_embeddings)
        #print("Document Similarity: ", doc_similarity)
        #print("Phrase Similarity: ", phrase_similarity_matrix)
        unselected = list(range(len(phrases)))
        select_idx = np.argmax(doc_similarity)
        #print("Select IDs : ", select_idx)
        selected = [select_idx]
        unselected.remove(select_idx)

        for _ in range(topk - 1):
            mmr_distance_to_doc = doc_similarity[unselected, :]
            mmr_distance_between_phrases = np.max(phrase_similarity_matrix[unselected][:, selected], axis=1)

            mmr = l * mmr_distance_to_doc - (1 - l) * mmr_distance_between_phrases.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]

            selected.append(mmr_idx)
            unselected.remove(mmr_idx)

        return selected, doc_similarity
    
if __name__ == '__main__':
    
        #mypath = '/home/shreya/Documents/CMSC723/yutaya/EmbedRank/embedrank/Hulth2003/Tokenized_Training'
        mypath = '/home/shreya/Documents/CMSC723/yutaya/EmbedRank/embedrank/Hulth2003/train_pos'
        write_path = '/home/shreya/Documents/CMSC723/yutaya/EmbedRank/embedrank/output_pos/'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        numKeys = 6
        
        flag_doc = False
        flag_phrase = False
        
        for file in onlyfiles:    
            if(file.endswith(".abstr_tok")):
                f = open(mypath+"/"+file,'r')
                f1 = f.read()
                document = list(f1.split('$'))
                '''
                bigrm = list(bigrams(f1.split()))
                temp = '#'.join(map(' '.join, bigrm))
                phrases = list(temp.split('#'))

                del phrases[-1]
                '''
                del document[1]
                f.close()
                
                
                
                #Get embedding of the document
                #print("Doc embedding")
                document_embeddings = get_sentence_embeddings(document, phrase = False, ngram = 'unigrams', model = 'wiki')
                
                #Get embedding of each candidate phrase
                #print("POS Phrase Emedding")
                phrases, phrase_embeddings = get_sentence_embeddings(document, phrase = True, ngram = 'unigrams', model = 'wiki')
                
                #print("Sans POS Phrase Embedding")
                #phrase_embeddings = get_sentence_embeddings(phrases, phrase = False, ngram = 'unigrams', model = 'wiki')
                
                flag_doc = True
                flag_phrase = True
                
            
            if(flag_doc and flag_phrase):
                flag_doc = False
                flag_phrase = False
                
                if(len(phrases) < numKeys):
                    numKeys = len(phrases)//2
                #EmbedRank++
                key_phrases, doc_sim = mmr1(phrases, phrase_embeddings, document_embeddings, 0.8, numKeys)
                #key_phrases, doc_sim = mmr1(phrases, phrase_embeddings, document_embeddings, 0.8, 6)
                
                #Store the results in a new file
                f = open(write_path+file+"_keyphrases.txt", "w")
                for i in key_phrases:
                    f.write(str(phrases[i])+",")
                f.close()
                
        #print("Key Phrases : " , key_phrases, "Similarity : ", doc_sim)
        
