from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras import optimizers
from keras.layers import LSTM
from keras.layers import Dropout
from scipy.sparse import *
import keras.callbacks
import itertools
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class RnnLstm():
 
  def __init__(self):
    ''' Model Data '''
    self.x_train = None
    self.y_train = None
    self.x_test = None
    self.y_test = None

   ''' Model variables '''
    self.model = Sequential()
    self.cells = 32
    self.output_dim = 2048
    self.dropout = 0.2
    self.recurrent_dropout = 0.2
    self.number_classes = 8
    self.verbose = 1
    self.batch_size = 64
    self.epochs = 40

    ''' Results '''
    self.score = None
    self.acc = None
    self.train_data_set_directory = None
    self.test_data_set_directory = None
    self.classes_directory = None
    self.unique_opcodes = None
    self.cols = None
    self.rows = None
 
  def set_max_features(self, value):
    self.max_features = value
 
  def set_output_dim(self, value):
    self.output_dim = value
 
  def set_number_classes(self, value):
    self.number_classes = value
 
  def set_batch_size(self, value):
    self.batch_size = value
 
  def set_epochs(self, value):
    self.epochs = value
 
  def set_train_data_set_directory(self, value):
    self.train_data_set_directory = value
 
  def set_test_data_set_directory(self, value):
    self.test_data_set_directory = value
 
  def set_classes_directory(self, value):
    self.classes_directory = value
 
  def build(self):
    directory = os.getcwd() + "/neural_network/summary.txt"
    summary_file = open(directory, 'w')
   
    self.set_classes()
    self.get_direccionary()
   
    self.x_train, self.y_train = self.load(self.train_data_set_directory)
    self.x_test, self.y_test = self.load(self.test_data_set_directory)
 
    max_features = self.cols
    #max_features = len(self.x_train)
   
    #self.x_train = self.x_train / np.linalg.norm(self.x_train)
    #self.x_test = self.x_test / np.linalg.norm(self.x_test)
 
   
    self.x_train = np.reshape(self.x_train,(len(self.x_train),self.cols))
    self.y_train = np.reshape(self.y_train,(len(self.y_train),1))
    self.x_test = np.reshape(self.x_test,(len(self.x_test),self.cols))
    self.y_test = np.reshape(self.y_test,(len(self.y_test),1))
   
    self.model.add(Embedding(max_features, output_dim = self.output_dim))
    self.model.add(LSTM(self.cells))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(8, activation='softmax'))
    adam = optimizers.Adam(lr=0.001)
    self.model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adam)
    self.model.summary()
   
    orig_stdout = sys.stdout
    sys.stdout = summary_file
    print(self.model.summary())
    sys.stdout = orig_stdout
    summary_file.close()
 
 
  def train(self):
   
    directory_fit = os.getcwd() + "/neural_network/fit.txt"
    directory_score = os.getcwd() + "/neural_network/score.txt"
 
    fit_file = open(directory_fit, 'w')
    score_file = open(directory_score, 'w')
   
    checkpointer = keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", verbose=1, monitor='val_f1', mode="max", save_best_only=True)
   
    orig_stdout = sys.stdout
    sys.stdout = fit_file
    print(self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[checkpointer]))
    sys.stdout = orig_stdout
    fit_file.close()
   
    #self.score = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
   
    y_pred = self.model.predict(self.x_test)
   
    y = np.argmax(y_pred, axis=1)
    #score_file.write("%s%s%s%s\n" % ("Test loss: ", a), "  Test accuracy: ", b)
   
    orig_stdout = sys.stdout
    sys.stdout = score_file
    result_score = f1_score(self.y_test, y, average='macro')
    print("%s%s\n" % (" f1_score_macro: ", result_score))
   
    result_score = precision_score(self.y_test, y, average='macro')  
    print("%s%s\n" % (" precision_macro: ", result_score))
 
 
    result_score = recall_score(self.y_test, y, average='macro')  
    print("%s%s\n" % (" recall_micro: ", result_score))
   
    print(accuracy_score(self.y_test, y))
   
    sys.stdout = orig_stdout
    score_file.close()
   
    #"confusion matrix"
    cnf_matrix = confusion_matrix(self.y_test, y)
    np.set_printoptions(precision=2)
    plot_title = 'Confusion matrix of rnn'
    self.plot_confusion_matrix(cnf_matrix, classes=self.classes,
                      title=plot_title)
    plt.savefig(plot_title)
   
 
  def accurate():
    self.score, self.acc = self.model.evaluate(self.x_test, self.y_test, batch_size = self.batch_size)
 
  def load(self, directory):
    labels = []
    data = []
    limit = 10
    for folder in os.listdir(directory):
      index = self.read_file(directory + "/" + folder + "/" + "index.txt")
      for file in index:
        frequency = defaultdict(int)
        file_path = directory + "/" + folder + "/" + "opcode/" + file + ".txt"
        opcodes = self.read_file(file_path)
        data_len = len(opcodes)
        #print(data_len)
        if data_len > limit :
            op = {el:0 for el in self.unique_opcodes}
            for val in opcodes:
              frequency[val] += 1
            for words in self.unique_opcodes:
              op[words] = frequency[words]
            data = data + [op.values()]
            labels = labels + [self.class_value(folder)]
    return data, labels
 
 
  def get_direccionary(self):
    train_unique_opcodes = self.load_files(self.train_data_set_directory)
    test_unique_opcodes = self.load_files(self.test_data_set_directory)
    self.unique_opcodes = list(set(train_unique_opcodes + test_unique_opcodes))
    self.cols = len(self.unique_opcodes)
 
  def set_classes(self):
    self.classes = self.read_file(self.classes_directory)
 
  def class_value(self, name):
    return self.classes.index(name)
 
  def load_files(self, directory):
    opcodes_vocabulary = []
    for folder in os.listdir(directory):
      index = self.read_file(directory + "/" + folder + "/" + "index.txt")
      for file in index:
        file_path = directory + "/" + folder + "/" + "opcode/" + file + ".txt"
        opcodes = self.read_file(file_path)
        opcodes_vocabulary = opcodes_vocabulary + list(set(opcodes))
    return opcodes_vocabulary
 
  def read_file(self, file_name):
    lines = [line.rstrip('\n') for line in open(file_name)]
    return lines
 
  def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
