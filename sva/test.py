from __future__ import division
import json
from rule_based import classify_verb as rules
from ml_based import classify_verb as ml
from pipeline import nlp

# test the quality
def test_quality(approach):
    """
    Calculate precision and recall of a solution.
    """
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
def test(s, approach):
    """
    Catch errors and print out the transform.
    """
    s_split = s.split()
    parsed_s = nlp(s)
    for i in xrange(len(parsed_s)):
        if parsed_s[i].tag_ == "VBZ":
            if approach(s, i) == 1:
                print str(1) + ":", " ".join(s_split[:i]), \
                    "[{}=>{}]".format(s_split[i], transform[s_split[i]]), \
                    " ".join(s_split[i + 1:]) + "\t({} {})".format(parsed_s[i], parsed_s[i].tag_)
            else:
                print str(0) + ":", s + "\t({} {})".format(parsed_s[i], parsed_s[i].tag_)


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
        test(u'We likes pizza with anchovy .', approach)
        test(u'Children like and cherishes her kindness and cooking skills .', approach)
        test(u'Some is watching the way she knits and loving it .', approach)
        test(u'Colorless green ideas sleeps furiously .', approach)
        test(u"Barry and Mary is just the cutest people .", approach)
        test(u"Barry and Mary , whom I met at the New Year 's party , is just the cutest people .", approach)
        test(u'There is two cats and a dog .', approach)

        # TN
        test(u'He likes pizza .', approach)
        test(u'The kid likes and cherishes her kindness and cooking skills .', approach)
        test(u'This one is watching the way she knits and loving it .', approach)
        test(u"Barry or Mary is just the cutest person .", approach)
