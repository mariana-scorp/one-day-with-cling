from __future__ import division
import random
from collections import defaultdict

if __name__ == '__main__':


    ##### Read the necessary data

    # "brown_pos.txt" - a data file. Size: ~ 27,500 sentences (~ 450,000 words).
    # "penn-tags.txt" - a list of Penn Treebank POS tags
    # "word-tags.txt" - a list of words and POS tags they may have extracted from Wiktionary

    # read train and test data
    with open("brown_pos.txt", "r") as f:
        data = f.readlines()
        random.seed(111)
        random.shuffle(data)
        test_part = len(data) // 10
        test_data = data[:test_part]
        train_data = data[test_part:]

    # read a list of POS tags
    with open("penn-tags.txt", "r") as f:
        tags = f.read().split()

    # read word-tag pairs
    word_tags = {}
    with open("word-tags.txt", "r") as f:
        for line in f.readlines():
            vals = line.strip().split()
            word_tags[vals[0]] = vals[1:]



    ##### Collect ngrams from the train set

    # a placeholder for all ngrams
    ngrams = defaultdict(int)

    # the total number of bigrams to be used for add-one smoothing
    total_bigrams = set()

    # collection of ngrams
    for sent in train_data:
        # sentence = "Chewie_NNP ,_, we_PRP 're_VBP home_RB !_!"
        # change to: [("Chewie", "NNP"), (",", ","), ("we", "PRP")...
        sent = [tuple(w_t.split("_")) for w_t in sent.split()]
        for i in range(len(sent)):
            # trigrams and bigrams of POS tags
            if i == 0:
                ngrams[("<S>", "<S>", sent[i][1])] += 1
                ngrams[("<S>", "<S>")] += 1
                total_bigrams.add(("<S>", "<S>"))
            elif i == 1:
                ngrams[("<S>", sent[i-1][1], sent[i][1])] += 1
                ngrams[("<S>", sent[i-1][1])] += 1
                total_bigrams.add(("<S>", sent[i-1][1]))
            else:
                ngrams[(sent[i-2][1], sent[i-1][1], sent[i][1])] += 1
                ngrams[(sent[i-2][1], sent[i-1][1])] += 1
                total_bigrams.add((sent[i-2][1], sent[i-1][1]))
            # word/tag statistics
            ngrams[sent[i][1]] += 1
            ngrams[sent[i][0] + "_" + sent[i][1]] += 1
        # trigrams and bigrams of POS tags at sentence end
        ngrams[(sent[i-1][1], sent[i][1], "</S>")] += 1
        ngrams[(sent[i-1][1], sent[i][1])] += 1
        total_bigrams.add((sent[i-1][1], sent[i][1]))


    # for example
    print "Love as noun: {}.".format(ngrams["love_NN"]),
    print "Love as verb: {}.".format(ngrams["love_VBP"] + ngrams["love_VB"])
    print "There are {} adjectives and {} adverbs in the corpus.".format(
        (ngrams["JJ"] + ngrams["JJR"] + ngrams["JJS"]),
        (ngrams["RB"] + ngrams["RBR"] + ngrams["RBS"]))
    print "Out of {} occurrences of (IN, DT), only {} times it was followed by NN.".format(
        ngrams[("IN", "DT")], ngrams[("IN", "DT", "NN")])



    ##### Hidden Markov Models

    def get_tags(word):
        """
        Return the list of acceptable POS tags for word.
        """
        tagset = []
        try:
            tagset += word_tags[word]
        except:
            pass
        try:
            tagset += word_tags[word.lower()]
        except:
            pass
        return tagset if tagset else tags

    def tag(sentence, print_info=True):
        """
        Get the most probable POS tag sequence for sentence.
        """

        # a placeholder for probabilities
        probs = defaultdict(int)
        probs[(-1, "<S>", "<S>")] = 1

        # a placeholder for most probable tags
        bp = {}

        # sentence length
        n = len(sentence)

        for i in range(n):
            # define possible POS tags in the trigram
            if i == 0:
                w_tags, u_tags = ["<S>"], ["<S>"]
            elif i == 1:
                w_tags, u_tags = ["<S>"], get_tags(sentence[i-1])
            else:
                w_tags, u_tags = get_tags(sentence[i-2]), get_tags(sentence[i-1])
            v_tags = get_tags(sentence[i])
            # go over all combinations of tags in the trigram: (w, u, v)
            # and remember the tag with the maximum probability
            for u in u_tags:
                for v in v_tags:
                    max_prob, max_tag = -1, None
                    for w in w_tags:
                        # Add-one (Laplace):
                        # add 1 in numerator
                        # add (V - the total number of possible (N-1)-grams) in denominator
                        val = probs[(i-1, w, u)] * \
                              ((ngrams[(w, u, v)] + 1) / (ngrams[(w, u)] + len(total_bigrams))) * \
                              ((ngrams[(sentence[0] + "_" + v)] + 1) / (ngrams[v] + len(tags)))
                        if val > max_prob:
                            max_prob, max_tag = val, w
                    probs[(i, u, v)] = max_prob
                    bp[(i, u, v)] = max_tag

        # calculating the final score at the sentence end
        max_prob, max_tags = -1, None
        for u in get_tags(sentence[n-2]):
            for v in get_tags(sentence[n-1]):
                val = probs[(n - 1, u, v)] * (ngrams[(u, v, "</S>")] + 1) / \
                      (ngrams[(u, v)] + len(total_bigrams))
                if val > max_prob:
                    max_prob, max_tags = val, [v, u]

        # get the most probable tags
        for i in range(n - 2, 0, -1):
            tag = bp[(i+1, max_tags[-1], max_tags[-2])]
            max_tags.append(tag)

        max_tags.reverse()

        if print_info:
            print
            print "Sentence: " + " ".join(sentence)
            print "The number of combinations we had to go through: {}.".format(len(probs.items()))
            print "The probability of the most probable sequence: {}.".format(max_prob)

        return max_tags



    ##### Trying it out

    # simple cases
    print tag("We 're home .".split())
    print tag("She charged off to the bedrooms .".split())
    print tag("Other countries , some of which I visited last month , have similar needs .".split())
    print tag("Colorless green ideas sleep furiously .".split())
    print tag("Wow , two hungry cats chased down the mouse to the corner and quickly ate it !".split())

    # ambiguous parts of speech
    print tag("All you need is love .".split())
    print tag("Maybe I 'm amazed at the way you love me all the time .".split())
    print tag("And never mind that noise you heard .".split())
    print tag("Dreams of war , dreams of liars , dreams of dragon 's fire and of things that will bite .".split())

    # impossible cases
    print tag("Time flies like an arrow .".split())
    print tag("I saw her duck with a telescope .".split())
    print tag("She is calculating .".split())
    print tag("We watched an Indian dance .".split())
    print tag("More lies ahead ...".split())



    ##### Evaluation

    tp, total = 0, 0
    for sent in test_data:
        # split the sentence
        sent = [tuple(w_t.split("_")) for w_t in sent.split()]
        # extract words and tags from the test sentence
        words, tags = [i[0] for i in sent], [i[1] for i in sent]
        # tag the sentence
        result = tag(words, print_info=False)
        # count correct POS tags
        for (i, j) in zip(tags, result):
            if i == j:
                tp += 1
        total += len(words)

    print
    print "The accuracy is {}%.".format(round(tp/total * 100, 2))

    # Without smoothing: 78%
    # With add-one smoothing: 87.58%

    # Reported results for Trigram HMM: 95.7%
    # Reported results for more complex HMM: 96.9%
