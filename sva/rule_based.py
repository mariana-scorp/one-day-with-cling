from __future__ import division
import spacy, json
from spacy.symbols import *


# dictionaries
PRP_PL = ["i", "you", "we", "they", "these", "those",
          "all", "both", "others", "some", "many", "few"]
PRP_SG = ["he", "she", "it", "this", "that", "each", "another"]


# define the pipeline
def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)
nlp = spacy.load('en', create_pipeline=custom_pipeline)


# classify VBZ

def has_plural_subject(w):
    "Check if the verb has an nsubj relation to a plural subject"

    for child in w.children:
        if child.dep == nsubj and \
                (child.tag_ in ["NNS", "NNPS"] or child.lemma_ in PRP_PL or \
                ((child.tag_ in ["NN", "NNP"] or child.lemma_ in PRP_SG) and \
                    "and" in [c.lemma_ for c in child.children])):
            return True

def classify_verb(sentence, id):
    "Decide if the verb needs to be changed. Return 1 if yes, 0 - if no."

    # sentence, given word and its head
    sentence = nlp(sentence)
    w = sentence[id]
    h = w.head

    # check if there is a plural subject among children
    if has_plural_subject(w):
        return 1

    # check if there is a plural subject among children of the head verb
    if w.dep in [aux, auxpass, conj] and has_plural_subject(h):
        return 1

    # check if there is a plural subject with expletive
    if expl in [c.dep for c in w.children]:
        for child in w.children:
            if child.dep in [dobj, attr] and child.tag_ in ["NNS", "NNPS"]:
                return 1

    return 0


# test the quality
def test_quality():
    tp, fn, fp, tn = 0, 0, 0, 0
    with open("test.txt", "r") as f:
        test_data = json.load(f)
    for i in test_data:
        rez = classify_verb(i[0][0], i[0][1])
        if rez == 1 and i[1] == 1:
            tp += 1
        elif rez == 0 and i[1] == 1:
            fn += 1
        elif rez == 1 and i[1] == 0:
            fp += 1
        else:
            tn += 1

    print "Precision: {}%.\nRecall: {}%.".format(
        round(tp / (tp + fp), 2), round(tp / (tp + fn), 2))


# read the transformations
transform = {}
with open("vbz_to_vbp.txt", "r") as f:
    for line in f.readlines():
        (w, t) = line.split()
        transform[w] = t


# test random sentences
def test(s, id):
    "Print out the transform."
    s_split = s.split()
    if classify_verb(s, id) == 1:
        print str(1) + ":", " ".join(s_split[:id]), \
            "[{}=>{}]".format(s_split[id], transform[s_split[id]]), \
            " ".join(s_split[id + 1:])
    else:
        print str(0) + ":", s