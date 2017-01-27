from __future__ import division
import json
from rule_based import classify_verb as rules
from ml_based import classify_verb as ml

# test the quality
def test_quality(approach):
    "Calculate precision and recall of a solution."
    tp, fn, fp, tn = 0, 0, 0, 0
    with open("test.txt", "r") as f:
        test_data = json.load(f)
    for i in test_data:
        rez = approach(i[0][0], i[0][1])
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
def test(s, id, approach):
    "Catch errors and print out the transform."
    s_split = s.split()
    if approach(s, id) == 1:
        print str(1) + ":", " ".join(s_split[:id]), \
            "[{}=>{}]".format(s_split[id], transform[s_split[id]]), \
            " ".join(s_split[id + 1:])
    else:
        print str(0) + ":", s


if __name__ == '__main__':

    print "RULE-BASED APPROACH:\n"
    test_quality(rules)

    # Precision: 0.6 %.
    # Recall: 0.9 %.

    print
    print "MACHINE LEARNING APPROACH:\n"
    test_quality(ml)

    # Precision: 0.83 %.
    # Recall: 0.83 %.

    print
    print

    for approach in [rules, ml]:
        j = "RULES" if approach == rules else "\nML CLASSIFIER"
        print j

        # TP
        test(u'We likes pizza with anchovy .', 1, approach)
        test(u'Children like and cherishes her kindness and cooking skills .', 3, approach)
        test(u'Some is watching the way she knits and loving it .', 1, approach)
        test(u'Colorless green ideas sleeps furiously .', 3, approach)
        test(u"Barry and Mary is just the cutest people .", 3, approach)
        test(u"Barry and Mary , whom I met at the New Year 's party , is just the cutest people .", 14, approach)
        test(u'There is two cats and a dog .', 1, approach)

        # TN
        test(u'He likes pizza .', 1, approach)
        test(u'The kid likes and cherishes her kindness and cooking skills .', 4, approach)
        test(u'This one is watching the way she knits and loving it .', 2, approach)
        test(u"Barry or Mary is just the cutest person .", 3, approach)
