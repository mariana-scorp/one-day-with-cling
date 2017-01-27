import random
import numpy as np
import spacy
from pandas import json
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from spacy.symbols import *
from cPickle import dump, load


pronouns = [u'i', u'you', u'we', u'they', u'these', u'those', u'all', u'both',
            u'others', u'some', u'many', u'few', u'he', u'she', u'it', u'this',
            u'that', u'each', u'another', u"one", u'them', u'theirs']

tag_lst = [u'WDT', u'JJ', u'WP', u'DT', u'NN', u'PRP', u'NNS', u'NNP', u'WRB',
           u'EX', u'NNPS', u'JJS', u'JJR', u'VBG', u'PDT', u'CD',
           u'VBZ', u'VBN', u'IN', u'CC', u'VBP', u'VBD', u'TO', u'RB']
# ^sentences with the weird pos of the subject are given in `wrong_parses.txt`

# define the pipeline
def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)
nlp = spacy.load('en', create_pipeline=custom_pipeline)

def has_subject(w):
    "Gets the tag and lemma of the subject."

    if sum(1 for x in w.children) > 0:
        for child in w.children:
            if child.dep == nsubj:
                return (child.tag_ , child.lemma_)
    return ('nostring', 'nostring')

# read the transformations
transform = {}
with open("vbz_to_vbp.txt", "r") as f:
    for line in f.readlines():
        (w, t) = line.split()
        transform[w] = t

def has_compound(w):
    "Checks if the subject has a compound."

    for child in w.children:
        if child.dep == nsubj and "and" in [c.lemma_ for c in child.children]:
            return True

def feature_extractor(sents, N):
    "Extracts feature for the sentence."

    # numpy lists of zeros for features
    subjects = np.zeros((N, 39), dtype=np.int)
    cc = np.zeros((N, 1), dtype=np.int)
    prns = np.zeros((N, 22), dtype=np.int)

    #parsing sentences
    all_parsed = [0] * N
    for k, s in enumerate(sents):
        parsed_s = nlp(s[0])
        all_parsed[k] = [parsed_s, s[1]]

    for j, sent in enumerate(all_parsed):

        id = sent[1]
        v = sent[0][id]
        h = v.head

        if v.tag_ == u'VBZ':

            # check if there is a plural subject among children
            tag, lemma = has_subject(v)
            if tag != 'nostring':
                if tag == u'PRP':
                    idx = pronouns.index(lemma)
                    prns[j][idx] = 1
                else:
                    idx = tag_lst.index(tag)
                    subjects[j][idx] = 1
                if has_compound(v):
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
                    if has_compound(v):
                        cc[j] = 1

            elif expl in [c.dep for c in v.children]:
                for child in v.children:
                    if child.dep in [dobj, attr]:
                        idx = tag_lst.index(child.tag_)
                        subjects[j][idx] = 1

    feats = np.hstack([subjects, cc, prns])

    return feats

def train_test():

    #defining classifier
    clf = svm.SVC(kernel='linear', C=1)
    # clf = RandomForestClassifier()

    #preparing data sets and classes
    with open("train.txt", "r") as f1, open("test.txt", "r") as f2:
        train_data = json.load(f1)
        test_data = json.load(f2)

    y_train = np.array(map(lambda x: x[1], train_data))
    y_test = np.array(map(lambda x: x[1], test_data))

    N_train = len(y_train)
    N_test = len(y_test)

    train_extr = map(lambda x: x[0], train_data)
    test_extr = map(lambda x: x[0], test_data)
    X_train = feature_extractor(train_extr, N_train)
    X_test = feature_extractor(test_extr, N_test)

    # print random.sample([[i for i, t in enumerate(x) if t]
    #                      for x, y in zip(X_train, y_train) if y == 0], 30)
    # print random.sample([[i for i, t in enumerate(x) if t]
    #                      for x, y in zip(X_train, y_train) if y == 1], 30)

    #train the classifier
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print 'Precision: {:0.2f}%'.format(precision_score(y_test, prediction))
    print 'Recall: {:0.2f}%'.format(recall_score(y_test, prediction))
    print 'Accuracy: {:0.2f}%'.format(accuracy_score(y_test, prediction))
    print 'F1 score: {:0.2f}%'.format(f1_score(y_test, prediction))

    #dump the classifier to the file
    output = open('sva_classifier.pkl', 'wb')
    dump(clf, output, -1)
    output.close()

# test random sentences
def test(s, id):
    "Classify a sentence and transform it if needed"

    input = open('sva_classifier.pkl', 'rb')
    model = load(input)
    input.close()

    X = feature_extractor([[s, id]], 1)

    s_split = s.split()
    if model.predict(X)[0] == 1:
        print str(1) + ":", " ".join(s_split[:id]), \
            "[{}=>{}]".format(s_split[id], transform[s_split[id]]), \
            " ".join(s_split[id + 1:])
    else:
        print str(0) + ":", s

if __name__ == '__main__':
    #1 step: train_test -> the model is being generated
    #2 step: test -> the sentence is being tested; outputs 0 or 1

    train_test()

    # TP
    test(u'We likes green eggs and ham .', 1)
    test(u'Children liked and cherishes green eggs and ham .', 3)
    test(u'These is loving green eggs and ham .', 1)
    test(u'Colorless green ideas sleeps furiously .', 3)
    test(u"Barry and Mary , whom I met at the New Year 's party , is just the cutest couple .", 14)
    test(u"Barry and Mary is just the cutest couple .", 3)
    test(u'There is two cats and a dog .', 1)
    # TN
    test(u'He likes green eggs and ham .', 1)
    test(u'The child liked and cherishes green eggs and ham .', 4)
    test(u'This one is loving green eggs and ham .', 2)