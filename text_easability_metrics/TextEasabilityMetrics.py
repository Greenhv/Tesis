from .LSA import LSA
from .StanfordNLP import StanfordCoreNLP, StanfordNLP
import nltk

class TextEasabilityMetrics:
  def __init__(self, path, port):
    self.nlp = StanfordNLP(language='es')
    print('TextEasabilityMetrics')

  def wordsBeforeMainVerb(self, senteceParsed):
    ''' This function will count the number of words before de main verb of the main clause '''
    numberOfWords = 0
    print(senteceParsed)
    print(type(senteceParsed))


    return numberOfWords


  def wordConcreteness(self, sentence):
    print("Word Concreteness")
  
  def syntaxisSimplicity(self, sentence):
    ''' This metric should use a Constituency Parser for Spanish provided by the Stanford library  '''
    senteceParsed = self.nlp.parse(sentence)

    # The sentence should be pre-procesing to extract its gramatical elements, but for the time we should use a noted corpus.
    numberOfWords = self.wordsBeforeMainVerb(senteceParsed)
    print("Syntactic Simplicity")

    return numberOfWords

  def verbCohesion(self, sentence):
    print("Verb Cohesion")

  def narrativity(self, text):
    print('Narrativity!')
  
  def temporalCohesion(self, text):
    print('Temporal cohesion!')

  def deepCohesion(self, text):
    print('Deep cohesion!')