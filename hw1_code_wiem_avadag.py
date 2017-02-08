"#!/usr/bin/python3"
from nltk import word_tokenize
from os import listdir
import math
import heapq
import random

wiem_avadag_d = "/home1/c/cis530/hw1/data/train/"

# 
# PRE-PROCESSING
#

def get_all_files(directory):
    return listdir(directory)

def standardize(rawexcerpt):
    return word_tokenize(rawexcerpt.lower())

def load_file_excerpts(filepath):
    with open(filepath, 'r') as f:
        return [standardize(line) for line in f.readlines()]


def load_directory_excerpts(dirpath):
    return flatten([load_file_excerpts(dirpath+f) for f in get_all_files(dirpath)])

def flatten(listoflists):
   return [elt for l in listoflists for elt in l]

#
# TF-IDF
#

def get_idf(corpus):
    d = {}
    n = len(corpus)
    for element in list(set(flatten(corpus))):
        word_count = sum(element in sublst for sublst in corpus)
        d[element] = math.log((n/word_count),math.e)
    return d


def get_idf_with_output_text(sample):
    f = open("./hw1_2-1a.txt")
    f2 = open("./hw1_2-1b.txt", 'w+')
    d = get_tf(sample)
    for line in f.readlines():
        word = line.split(None, 1)[0]
        value = float(line.split(None, 1)[1])
        w = value/d[word]
        f2.write(word+"\t"+str(w)+"\n")


def get_tf(sample):
    s = flatten(sample)
    word_freqs = {}
    for word in s:
        word_freqs[word] = word_freqs.get(word,0) + 1
    return word_freqs


def get_number_of_excerpts(dirpath):
    return len(load_directory_excerpts(dirpath))


def get_tfidf(tf_dict, idf_dict):
    d = {}
    for element in tf_dict.keys():
        d[element] = tf_dict[element] * idf_dict[element]
    return d


def get_tfidf_weights_topk(tf_dict, idf_dict, k):
    f1 = open('./corpustfidf.txt', 'w+')
    dictionary = get_tfidf(tf_dict,idf_dict)
    s = heapq.nlargest(k,dictionary,key=dictionary.get)
    for e in s:
        f1.write(e+"\t"+str(dictionary[e])+"\n")
    return s


def get_tfidf_topk(sample,corpus,k):
    tf = get_tf(sample)
    idf = get_idf(corpus)
    get_tfidf_weights_topk(tf,idf,k)

#
# Mutual Information
#

def get_word_freqs(text):
    word_freqs = {}
    for word in text:
        word_freqs[word] = word_freqs.get(word,0) + 1
    return word_freqs

# returns a dict with key=w, value=p(w)
def get_word_probs(sample):
    word_freqs = get_word_freqs(sample)
    word_probs = {}
    for word in word_freqs:
        word_probs[word] = word_freqs[word]/len(sample)
    return word_probs

def get_mi_from_probs(sample_probs, corpus_probs):

    mi = {}
    for word in sample_probs.keys():
        s_prob = sample_probs.get(word,0)
        c_prob = corpus_probs.get(word,0)
        if(c_prob==0):
            mi[word]=0
        else:
            mi[word]=math.log((s_prob/c_prob),math.e)
    return mi

def get_mi(sample, corpus):
    word_freqs = get_word_freqs(corpus)
    s_word_probs = get_word_probs(sample)
    c_word_probs= get_word_probs(corpus)
    mi = {}
    for word in sample:
        s_prob = s_word_probs[word]
        c_prob = c_word_probs[word]
        if(word_freqs[word] >= 5):
            mi[word]=math.log(s_prob/c_prob,math.e)
    return mi

#
# REQUIRED FUNCTION
#

def get_mi_topk(sample, corpus, k):
    mi = get_mi(sample, corpus)
    return heapq.nlargest(k, mi.items(), key=lambda s: s[1])

#
# PRECISION AND RECALL
#

def intersect(l1,l2):
    return list(set(l1) & set(l2))

def get_precision(l1,l2):
    return len(intersect(l1,l2)) / len(l1)

def get_recall(l1,l2):
    return len(intersect(l1,l2)) / len(l2)

#
# COSINE SIMILARITY
#

def dot(l1, l2):
    s = 0
    for i in range (0, len(l1)):
        s += l1[i]*l2[i]
    return s

def norm(l):
    s = 0
    for val in l:
        s += val*val
    return math.sqrt(s)

#
# REQUIRED FUNCTION
#
def cosine_sim(l1, l2):
    return dot(l1,l2) / (norm(l1)*norm(l2))

#
# LABELING NEW EXCERPTS
#
"Get list of words from the call get_tfidf_topk(corpus,corpus,1000)"
def get_word_list():
    list = []
    f = open("./corpustfidf.txt")
    for line in f.readlines():
        word = line.split(None, 1)[0]
        list.append(word)
    return list


def create_feature_space(wordlist):
    list_of_random = random.sample(range(0, len(wordlist)), len(wordlist))
    d = {}
    i = 0
    for word in wordlist:
        d[word] = list_of_random[i]
        i += 1
    return d


def get_idf_dict(filepath):
    dict = {}
    f = open(filepath, 'r')
    for line in f.readlines():
        word = line.split(None, 1)[0]
        value = float(line.split(None, 1)[1])
        dict[word] = value
    return dict


def vectorize_tfidf(feature_space, idf_dict, sample):
    default = 0
    tf_dict = get_tf(sample)
    vector = [0] * 1000
    for word in feature_space.keys():
        i = feature_space[word]
        vector[i] = tf_dict.get(word, default) * idf_dict.get(word,default)
    return vector


def get_section_representations(dirname, idf_dict, feature_space):
    files = get_all_files(dirname)
    vectors = {}
    for f in files:
        section = load_file_excerpts(dirname+f)
        v = vectorize_tfidf(feature_space, idf_dict, section)
        vectors[f[:-4]] = v
    return vectors


def predict_class(excerpt, representation_dict, feature_space, idf_dict):
    excerpt_vector = vectorize_tfidf(feature_space, idf_dict, [excerpt])
    cos_sim = {}
    for section in representation_dict:
        sim = cosine_sim(excerpt_vector, representation_dict[section])
        cos_sim[sim] = section
    max_sim = max(cos_sim.keys(),key=float)
    return cos_sim[max_sim]
    
#
# REQUIRED FUNCTION
#

def label_sents(excerptfile, outputfile):
    # get_tfidf_topk(corpus,corpus,1000) #writes to corpustfidf.txt
    excerpts = load_file_excerpts(excerptfile)
    f = open(outputfile, 'w+')
    feature_space = create_feature_space(get_word_list())
    idf_dict = get_idf_dict("./hw1_2-1b.txt")
    sect_rep = get_section_representations(d,idf_dict,feature_space)
    
    for excerpt in excerpts:
        f.write(predict_class(excerpt,sect_rep,feature_space,idf_dict)+"\n")

#
# CLUSTERING
#
def vectorize_mi(feature_space, word_probs, sample):

    default = 0
    mi = get_mi_from_probs(get_word_probs(sample), word_probs)
    vector = {}

    for word in feature_space:
        i = feature_space[word]
        vector[i] = mi.get(word,default)
    return vector

#
# REQUIRED FUNCTIONS
#
def prepare_cluto_tfidf(samplefile, labelfile, matfile, corpus):
    idf_dict = get_idf_dict("./hw1_2-1b.txt")
    labels_list = get_word_list()
    feature_space = create_feature_space(labels_list)
    f2 = open(labelfile, 'w+')
    for word in labels_list:
        f2.write(word+"\n")
    
    dictionary_excerpt_to_vec = {}
    sample = load_file_excerpts(samplefile)

    for i in range(len(sample)):
        dictionary_excerpt_to_vec[i] = vectorize_tfidf(feature_space,idf_dict,[sample[i]])

    f = open(matfile, 'w+')
    f.write(str(len(dictionary_excerpt_to_vec.keys())) + " " + str(len(dictionary_excerpt_to_vec[0])) + "\n")
    for index in range(len(dictionary_excerpt_to_vec)):
        for i in range(len(dictionary_excerpt_to_vec[index])):
            f.write(str(dictionary_excerpt_to_vec[index][i]) + " ")
        f.write("\n")
    return


def prepare_cluto_mi(samplefile, labelfile, matfile, corpus):
    labels = get_word_list()
    
    lf = open(labelfile, 'w+')
    for label in labels:
        lf.write(label+'\n')

    samples = load_file_excerpts(samplefile)
    feature_space = create_feature_space(labels)
    word_probs = get_word_probs(corpus)

    mf = open(matfile, 'w+')
    mf.write(str(len(samples))+' '+'1000\n')
    for sample in samples:
        vector = vectorize_mi(feature_space, word_probs, sample)
        for i in range(0,999):
            mf.write(str(vector[i]) + ' ')
        mf.write(str(vector[999]) + '\n')


if __name__ == "__main__":
    print("running")
