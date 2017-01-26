import random, time
from aux.dependencycorpusreader import DependencyCorpusReader
from aux.transitionparser import TransitionParser
from aux.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
from aux.dependencygraph import DependencyGraph

SIZE = 200

if __name__ == '__main__':

    # read train and test data
    data = DependencyCorpusReader("", ['en-universal.conll']).parsed_sents()
    random.seed(111)
    random.shuffle(data)
    subdata = data[:SIZE]
    test_part = len(subdata) // 10
    test_data = subdata[:test_part]
    train_data = subdata[test_part:]

    print "The data is ready."


    # training

    print "Starting the training..."
    start = time.time()

    # train the parser
    tp = TransitionParser(Transition, FeatureExtractor)
    tp.train(train_data)
    tp.save("english-" + str(SIZE) + "model")

    stop = time.time()
    print "Time elapsed: {}s.".format(int(stop - start))

    # load the model
    tp = TransitionParser.load("english-" + str(SIZE) + "model")


    # parsing an arbitrary sentence

    sentence = DependencyGraph.from_sentence('This is a test.')
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    sentence = DependencyGraph.from_sentence('My cat chased the little mouse and ate it!')
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    # ambiguous cases
    sentence = DependencyGraph.from_sentence("Wanted: a nurse for a baby about twenty years old.")
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    sentence = DependencyGraph.from_sentence("I shot an elephant in my pajamas.")
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    sentence = DependencyGraph.from_sentence("I once saw a deer riding my bicycle.")
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    sentence = DependencyGraph.from_sentence("I'm glad I'm a man, and so is Lola.")
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

    sentence = DependencyGraph.from_sentence("We eat pizza with anchovy.")
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')


    # testing

    # parse test data
    parsed = tp.parse(test_data)

    # write test data
    with open('test.conll', 'w') as f:
       for p in parsed:
           f.write(p.to_conll(10).encode('utf-8'))
           f.write('\n')

    # evaluate
    ev = DependencyEvaluator(test_data, parsed)
    print "UAS: {} \nLAS: {}".format(*ev.eval())



# 200 training examples
# Time elapsed: 140s.
# UAS: 0.682051282051
# LAS: 0.638461538462

# 2000 training examples
# Time elapsed: 5745s. ~1.5hr
# UAS: 0.822968580715
# LAS: 0.791332611051

# 4500 training examples
# Time elapsed: 29054s. ~ 8hrs
# UAS: 0.860304659498
# LAS: 0.838082437276
