import json
import numpy as np
from sklearn import svm
from sklearn.metrics import *
from spacy.symbols import *
from cPickle import dump, load
from pipeline import nlp


pronouns = [u'i', u'you', u'we', u'they', u'these', u'those', u'all', u'both',
            u'others', u'some', u'many', u'few', u'he', u'she', u'it', u'this',
            u'that', u'each', u'another', u"one", u'them', u'theirs', u'n']

tag_lst = [u'WDT', u'JJ', u'WP', u'DT', u'NN', u'PRP', u'NNS', u'NNP', u'WRB',
           u'EX', u'NNPS', u'JJS', u'JJR', u'VBG', u'PDT', u'CD', u'VB',
           u'VBZ', u'VBN', u'IN', u'CC', u'VBP', u'VBD', u'TO', u'RB', u'ADD']

NAME = 'sva_classifier.pkl'

def has_subject(w):
    """
    Get the tag and the lemma of the subject.
    """
    if sum(1 for x in w.children) > 0:
        for child in w.children:
            if child.dep == nsubj:
                return (child.tag_, child.lemma_)
    return ('nostring', 'nostring')

def has_conjunct(w):
    """
    Check if the subject has a conjunct.
    """
    for child in w.children:
        if child.dep == nsubj and "and" in [c.lemma_ for c in child.children]:
            return True

def feature_extractor(sents):
    """
    Extracts feature for the sentence.
    """

    N = len(sents)
    # numpy lists of zeros for features
    subjects = np.zeros((N, 39), dtype=np.int)
    cc = np.zeros((N, 1), dtype=np.int)
    prns = np.zeros((N, len(pronouns)), dtype=np.int)

    #parsing sentences
    all_parsed = [0] * N
    for k, s in enumerate(sents):
        parsed_s = nlp(s[0])
        all_parsed[k] = [parsed_s, s[1]]

    for j, sent in enumerate(all_parsed):

        id = sent[1]
        v = sent[0][id]
        h = v.head

        # check if there is a plural subject among children
        tag, lemma = has_subject(v)
        if tag != 'nostring':
            if tag == u'PRP':
                idx = pronouns.index(lemma)
                prns[j][idx] = 1
            else:
                idx = tag_lst.index(tag)
                subjects[j][idx] = 1
            if has_conjunct(v):
                cc[j] = 1

        elif v.dep in [aux, auxpass, conj]:
            tag, lemma = has_subject(h)
            if tag != 'nostring':
                if tag == u'PRP':
                    idx = pronouns.index(lemma)
                    prns[j][idx] = 1
                else:
                    idx = tag_lst.index(tag)
                    subjects[j][idx] = 1
                if has_conjunct(v):
                    cc[j] = 1

        elif expl in [c.dep for c in v.children]:
            for child in v.children:
                if child.dep in [dobj, attr]:
                    idx = tag_lst.index(child.tag_)
                    subjects[j][idx] = 1

    feats = np.hstack([subjects, cc, prns])

    return feats

def train():

    #defining a classifier
    clf = svm.SVC(kernel='linear', C=1)

    print "Training..."

    #preparing data sets and classes
    with open("train.txt", "r") as f1: #, open("test.txt", "r") as f2:
        train_data = json.load(f1)
        # test_data = json.load(f2)

    y_train = np.array(map(lambda x: x[1], train_data))
    # y_test = np.array(map(lambda x: x[1], test_data))

    train_extr = map(lambda x: x[0], train_data)
    # test_extr = map(lambda x: x[0], test_data)
    X_train = feature_extractor(train_extr)
    # X_test = feature_extractor(test_extr)

    #train the classifier
    clf.fit(X_train, y_train)
    print "Done."

    # # test the classifier
    # prediction = clf.predict(X_test)
    #
    # print 'Precision: {:0.2f}%'.format(precision_score(y_test, prediction))
    # print 'Recall: {:0.2f}%'.format(recall_score(y_test, prediction))
    # print 'Accuracy: {:0.2f}%'.format(accuracy_score(y_test, prediction))
    # print 'F1 score: {:0.2f}%'.format(f1_score(y_test, prediction))

    #dump the classifier to the file
    output = open(NAME, 'wb')
    dump(clf, output, -1)
    output.close()

def classify_verb(s, id):
    """
    Decide if the verb needs to be changed. Return 1 if yes, 0 - if no.
    """
    input = open(NAME, 'rb')
    model = load(input)
    input.close()
    X = feature_extractor([[s, id]])
    return model.predict(X)

if __name__ == '__main__':
    train()
