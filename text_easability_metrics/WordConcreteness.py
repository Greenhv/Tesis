from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

class WordConcreteness:
  def __init__(self):
    print('word concreteness')
  
  def removeSpecialCharacters(self, strWord):
    return ''.join(character for character in strWord if character.isalnum())

  def getWordNetTag(self, tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('S'):
        return wordnet.ADJ_SAT
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    else:
        return ''

  def getWordConcretenessInSentece(self, sentece):
    concretenessLvl = 0
    sentenceToken = nltk.word_tokenize(sentence)
    posTags = nltk.pos_tag(sentenceToken)
    for taggedWord in  posTags:
      word = taggedWord[0]
      tag = taggedWord[1]pos_tag
      if (getWordNetTag(tag)):
        for ss in wordnet.synsets(word, getWordNetTag(tag), lang='spa'):
            hyperyms = ss.hypernym_paths()[0]
            if (len(hyperyms)) > 1:
              category = ss.hypernym_paths()[0][1]
              concretenessLvl += 1 if "physical" in category.name() else 0
      return concretenessLvl

  def getWordConcretenessInText(self, currentText):
    textTotal = 0
    for paragraph in currentText:
      sentenceCount = 0
      for sentence in paragraph:
        sentenceCount += self.getWordConcretenessInSentece(sentece)
      textTotal += sentenceCount

    return textTotal