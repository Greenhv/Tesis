from .LSA import LSA
from .StanfordNLP import StanfordCoreNLP, StanfordNLP
from nltk.tree import Tree

class TextEasabilityMetrics:
  def __init__(self, path, port):
    '''
      All the methods in the class will work with only sentences. We think of this functions as granular
      functions which can be added to any procedure you want to build
    '''
    self.nlp = StanfordNLP(language='es')
    print('TextEasabilityMetrics')

  def isSpecialChar(self, leave):
    return not leave.isalnum()

  def getMainVerbRecursive(self, treePP, currentDeep, wordsCount, isMainVerbFound):
    wordsCountPP = wordsCount

    if(isinstance(treePP, Tree) and len(treePP) == 1) and treePP.height() == 2 and not isMainVerbFound:
#         print(treePP)
#         print(currentDeep)
      verbFound = treePP.label().startswith("v")
      wordsCountPP = wordsCountPP + (1 if not verbFound else 0)

      return currentDeep, wordsCountPP, verbFound
    else:
      partialDeep = 10000
      for index in range(len(treePP)):
        subTreeP = treePP[index]
        deepPP, wordsCountPP, mainVerFound = self.getMainVerbRecursive(subTreeP, currentDeep + 1, wordsCountPP, isMainVerbFound);
        if (mainVerFound and deepPP < partialDeep): partialDeep = deepPP
      return partialDeep, wordsCountPP, mainVerFound

    return currentDeep, wordsCount, isMainVerbFound

  def wordsBeforeMainVerb(self, sentenceTree):
    ''' This function will count the number of words before de main verb of the main clause '''
    currentDeep = 10000
    numOfWords = 0
    indexFound  = -1
    arrOfNumWords = list()

    for j in range(len(sentenceTree)):
        print('-------------------------------------------------------------------------------------')
        print(j)
        print(sentenceTree[j])
        # print('leaves', len(sentenceTree[j].leaves()))
        isFound = False
        newDeep, numWordsBefore, isFound = self.getMainVerbRecursive(sentenceTree[j], 1, 0, isFound)

        if (not isFound):
            numWordsBefore = len([leave for leave in sentenceTree[j].leaves() if not self.isSpecialChar(leave)])

        arrOfNumWords.append(numWordsBefore)

        if (isFound and newDeep < currentDeep):
            indexFound = j
            currentDeep = newDeep

        print(isFound, numWordsBefore, newDeep, currentDeep, indexFound)
        print('====================================================================================')

    print(arrOfNumWords, indexFound)
    numOfWords =  sum(arrOfNumWords[:indexFound]) if indexFound >= 0 else sum(arrOfNumWords)
    return numOfWords

  def wordConcreteness(self, sentence):
    print("Word Concreteness")
  
  def syntaxisSimplicity(self, sentence):
    '''
      This metric uses the Constituency Parser for Spanish provided by the Stanford CoreNLP Library v3.8.1.
      Also, this implementation is one of three metrics, the other two are are already implemented in
      the library Coh-Metrix-Esp (https://github.com/andreqi/coh-metrix-esp)
    '''
    sentenceTreeStr = self.nlp.parse(sentence)
    sentenceTree = Tree.fromstring(sentenceTreeStr)
    sentenceTree = sentenceTree[0] # We remove the ROOT element as the Tree head

    # The sentence should be pre-procesing to extract its gramatical elements, but for the time we should use a noted corpus.
    numberOfWords = self.wordsBeforeMainVerb(sentenceTree)
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