from spacy.symbols import *
from pipeline import nlp

# dictionaries
PRP_PL = ["i", "you", "we", "they", "these", "those",
          "both", "others", "some", "many", "few"]
PRP_SG = ["he", "she", "it", "this", "that", "each", "another", "one"]


# classify VBZ

def has_plural_subject(w):
    """
    Check if the verb has an nsubj relation to a plural subject.
    """
    for child in w.children:
        if child.dep == nsubj and \
                (child.tag_ in ["NNS", "NNPS"] or child.lemma_ in PRP_PL or \
                ((child.tag_ in ["NN", "NNP"] or child.lemma_ in PRP_SG) and \
                    "and" in [c.lemma_ for c in child.children])):
            return True

def classify_verb(s, id):
    """
    Decide if the verb needs to be changed. Return 1 if yes, 0 - if no.
    """

    # sentence, given word and its head
    s = nlp(s)
    w = s[id]
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
