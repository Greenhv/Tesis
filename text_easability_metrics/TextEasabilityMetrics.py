from .LSA import LSA
from .StanfordNLP import StanfordCoreNLP, StanfordNLP
from nltk.tree import Tree

class TextEasabilityMetrics:
  def __init__(self, host, port):
    '''
      All the methods in the class will work with only sentences. We think of this functions as granular
      functions which can be added to any procedure you want to build
    '''
    self.nlp = StanfordNLP(port=port, language='es')
    print('TextEasabilityMetrics')

  def isSpecialChar(self, leave):
    return not leave.isalnum()

  def getMainVerbRecursive(self, treePP, currentDeep, wordsCount, isMainVerbFound):
    wordsCountPP = wordsCount

    if(isinstance(treePP, Tree) and len(treePP) == 1) and treePP.height() == 2 and not isMainVerbFound:
      # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
      verbFound = treePP.label().startswith("v")
      wordsCountPP = wordsCountPP + (1 if not verbFound else 0)
      wordFound = treePP[0] if verbFound else ''
      # print(treePP, currentDeep, verbFound, wordsCount)

      return currentDeep, wordsCountPP, verbFound, wordFound
    else:
      partialDeep = 10000
      currentWord = ''
      for index in range(len(treePP)):
        subTreeP = treePP[index]
        deepPP, wordsCountPP, mainVerFound, wordFound = self.getMainVerbRecursive(subTreeP, currentDeep + 1, wordsCountPP, isMainVerbFound);

        if (mainVerFound and deepPP < partialDeep):
          partialDeep = deepPP
          currentWord = wordFound

        # print('............................................................................................................')
        # print('In recursive', f'tree: {subTreeP}', f'index {index}', f'len {len(treePP)}', f'deep: {partialDeep}', deepPP, wordsCount, mainVerFound, currentWord, wordFound)
      return partialDeep, wordsCountPP, currentWord != '', currentWord 

  def wordsBeforeMainVerb(self, sentenceTree):
    ''' This function will count the number of words before de main verb of the main clause '''
    currentDeep = 10000
    numOfWords = 0
    indexFound  = -1
    arrOfNumWords = list()
    currentWord = ''

    for j in range(len(sentenceTree)):
        # print('-------------------------------------------------------------------------------------')
        # print(j)
        # print(sentenceTree[j])
        # print('leaves', len(sentenceTree[j].leaves()))
        isFound = False
        newWord = ''
        newDeep, numWordsBefore, isFound, newWord = self.getMainVerbRecursive(sentenceTree[j], 1, 0, isFound)

        if (not isFound):
            numWordsBefore = len([leave for leave in sentenceTree[j].leaves() if not self.isSpecialChar(leave)])

        arrOfNumWords.append(numWordsBefore)

        if (newWord != '' and newDeep < currentDeep):
            indexFound = j
            currentDeep = newDeep
            currentWord = newWord

        # print('***************************************************************************')
        # print('After Recursive: ', isFound, numWordsBefore, newDeep, currentDeep, indexFound, currentWord, newWord)
        # print('====================================================================================')

    # print(arrOfNumWords, indexFound, currentWord)
    # numOfWords = sum(arrOfNumWords[:indexFound]) if indexFound >= 0 else sum(arrOfNumWords)
    treeLeaves = sentenceTree.leaves()
    leafIndex = treeLeaves.index(currentWord) if currentWord != '' else 0
    usefulLeaves = treeLeaves[:leafIndex]
    numOfWords = len([leaf for leaf in usefulLeaves if not self.isSpecialChar(leaf)])
    # print(usefulLeaves, len(usefulLeaves))

    return numOfWords

  def wordConcreteness(self, sentence):
    print("Word Concreteness")
  
  def syntaxisSimplicity(self, sentence):
    '''
      This metric uses the Constituency Parser for Spanish provided by the Stanford CoreNLP Library v3.8.1.
      Also, this implementation is one of three metrics, the other two are are already implemented in
      the library Coh-Metrix-Esp (https://github.com/andreqi/coh-metrix-esp)
    '''
    # print(sentence)
    # print("========================================================")
    sentenceTreeStr = self.nlp.parse(sentence)
    sentenceTree = Tree.fromstring(sentenceTreeStr)
    sentenceTree = sentenceTree[0] # We remove the ROOT element as the Tree head

    # The sentence should be pre-procesing to extract its gramatical elements, but for the time we should use a noted corpus.
    numberOfWords = self.wordsBeforeMainVerb(sentenceTree)
    # print("Syntactic Simplicity")

    return numberOfWords

  def verbCohesion(self, sentence):
    print("Verb Cohesion")

  def narrativity(self, text):
    print('Narrativity!')
  
  def temporalCohesion(self, text):
    print('Temporal cohesion!')

  def deepCohesion(self, text):
    print('Deep cohesion!')