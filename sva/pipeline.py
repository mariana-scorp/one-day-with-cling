import spacy
from spacy.tokens import Doc

# a whitespace tokenizer
class WhitespaceTokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

# define the pipeline
def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)

nlp = spacy.load('en', create_pipeline=custom_pipeline, create_make_doc=WhitespaceTokenizer)
