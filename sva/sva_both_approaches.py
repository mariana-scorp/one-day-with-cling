import spacy, json
from spacy.symbols import *

from rule_based import test as test_rb
from rule_based import test_quality as quality_rb
from ml_sva import test as test_ml
from ml_sva import train_test as quality_ml

if __name__ == '__main__':
    print "RULE BASED APPROACH:\n"
    quality_rb()
    print
    print "MACHINE LEARNING APPROACH:\n"
    quality_ml()
    print
    print

    for test in [test_rb, test_ml]:
        j = "RULES" if test == test_rb else "\nML CLASSIFIER"
        print j

        # TP
        test(u'We likes pizza .', 1)
        test(u'Children liked and cherishes her kindness .', 3)
        test(u'These is loving to swim .', 1)
        test(u'Colorless green ideas sleeps furiously .', 3)
        test(u"Barry and Mary , whom I met at the New Year 's party , is just the cutest couple .", 14)
        test(u"Barry and Mary is just the cutest couple .", 3)
        test(u'There is two cats and a dog .', 1)

        # TN
        test(u'He likes pizza .', 1)
        test(u'Children liked and cherished her kindness .', 4)
        test(u'This one is loving to swim .', 2)