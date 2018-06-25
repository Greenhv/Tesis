import re
from gensim import corpora, models, similarities
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

class VerbCohesion:
  def __init__(self, nlp):
    ''' This metric uses the Stanford CoreNLP Library v3.8.1. '''
    self.nlp = nlp

  def extractVerbsFromText(self, text):
    pos_tags = self.nlp.pos(text)
    verbs = [pos_tag[0] for pos_tag in pos_tags if pos_tag[1].startswith('v')]
    return verbs
  
  def clean_text(self, text):
    return extractVerbsFromText(text)
  
  def clean_texts(self, data):
    tokenized_data = []
    for text in data:
      tokenized_data.append(clean_text(text))
    return tokenized_data
  
  def generateLSAModel(self, sentences_or_words, isClean=False):
    if (isClean):
      clean_data = sentences_or_words
    else:
      clean_data = clean_texts(sentences)

    # Build a Dictionary - association word to numeric id
    dictionary = corpora.Dictionary(clean_data)

    # Transform the collection of texts to a numerical form
    corpus = [dictionary.doc2bow(text) for text in clean_data]

    # Build the LSI model
    lsi_model = models.LsiModel(corpus=corpus, num_topics=2, id2word=dictionary)
    
    return lsi_model, dictionary, corpus