# required imports
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

# Class that centralizes all the functionality required by our Part-of-Speech Tagger. It allows storage of the datasets to be employed
# in training, validation and testing, building models following the proposed architecture, training them, testing their performance,
# retrieving their predictions, and performing tagging over individual, user-defined sentences. Additionally, it allows saving and loading
# previously trained models.
class POS_Tagger:

  # constructor of the class; it requires a training dataset to be provided, while validation and testing datasets are optional
  # it's important to mention that the dataset format that this function expects is the one produced by using the preprocessing functions
  # available on the preprocess.py file
  def __init__(self, dataset_train=None, dataset_val=None, dataset_test=None):
    assert dataset_train is not None, "You must provide at least a training dataset!" # check if a train dataset has been provided
    
    # we create and store the tensor equivalents of the input/target series on the original dataset, and build a tensorflow dataset
    # from those tensors (also defining the batch size of said dataset).
    self.train_X=tf.convert_to_tensor(dataset_train.iloc[:, 0])
    self.train_Y=tf.convert_to_tensor(dataset_train.iloc[:, 1].to_list())
    self.train_ds=tf.data.Dataset.from_tensor_slices((self.train_X, self.train_Y))
    self.train_ds=self.train_ds.batch(64)
    
    # we repeat the process for validation and test, if they are available
    if dataset_val is not None:
      self.val_X=tf.convert_to_tensor(dataset_val.iloc[:, 0])
      self.val_Y=tf.convert_to_tensor(dataset_val.iloc[:, 1].to_list())
      self.val_ds=tf.data.Dataset.from_tensor_slices((self.val_X, self.val_Y))
      self.val_ds=self.val_ds.batch(64)
      
    if dataset_test is not None:
      self.test_X=tf.convert_to_tensor(dataset_test.iloc[:, 0])
      self.test_Y=tf.convert_to_tensor(dataset_test.iloc[:, 1].to_list())
      self.test_ds=tf.data.Dataset.from_tensor_slices((self.test_X, self.test_Y))
      self.test_ds=self.test_ds.batch(64)
    
    # at the beginning, the model is non-existent, and its training history too. Keep in mind this history is not always equivalent to the
    # History object returned by tensorflow after finishing training. We will just use this variable to check if the model has been trained,
    # and to know that it suffices to check that it is not None.
    self.model = None
    self.history = None
      
  # auxiliary, private method that retrieves a dictionary key by supplying a value
  # we use it when tagging user sentences: the prediction of the model will be a vector of labels, but we want the tags to be shown to the user
  # and thus we need to invert the encoding
  def __key_by_value(self, dictionary, target):
    for key, value in dictionary.items():
        if value == target:
            return key
    return None

  # method to build and compile the model, following the proposed architecture. The user can adjust the output dimensions of the embedding layer, and
  # the LSTM layer; and the number of possible labels. Their default values offer good enough results, in any case.
  def build(self, embedding_dim=30, LSTM_dim=128, nlabels=18):
    # we define the text vectorizer, and adjust it to the training inputs; it's important to set the standardization to just lowercasing, as
    # the default value would eliminate punctuation and we don't want that. The fixed output sequence will already pad the inputs to 128
    # elements, just like we manually did for the targets
    text_vectorizer=keras.layers.TextVectorization(output_mode='int', output_sequence_length=128, standardize='lower', split='whitespace')
    text_vectorizer.adapt(self.train_X)
    
    # we define the basic architecture
    inputs = keras.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = keras.layers.Embedding(text_vectorizer.vocabulary_size(), embedding_dim, mask_zero=True)(x) # never remove mask_zero=True, or the accuracy will be misleading
    x = keras.layers.LSTM(LSTM_dim, return_sequences=True)(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(nlabels, activation='softmax'))(x)
    self.model = keras.models.Model(inputs=inputs, outputs=outputs, name="POS_Tagging")
    self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy']) # we use sparse_categorical_crossentropy so that output does not require one-hot encoding
    
  # method to perform the model fitting
  # a model must have been either built, or imported, and the user may select the number of epochs (although the default works well)
  def train(self, num_epochs=10):
    assert self.model is not None, "You must build your model, by calling build(), before performing any training" #check that a model exists for this tagger
    
    # we perform the fitting, passing a validation dataset if it's present
    # in any case, we story the training history, so that it stays accesible and, more importantly, works as a check of whether or not we've trained the model
    if self.val_ds is not None:
      self.history = self.model.fit(self.train_ds, epochs=num_epochs, validation_data=self.val_ds)
    else:
      self.history = self.model.fit(self.train_ds, epochs=num_epochs)
      
  # method to test the accuracy of a previously trained model
  # it allows the user to supply his own test dataset, but keep in mind that the dataset provided should already be preprocessed in the manner
  # we applied at the constructor (that is, transformed into tensors, and those tensors into a tf.Dataset)
  # if a dataset is not provided, it uses the one given at creation (unless it is not present, in which case the call fails)
  def test(self, test_data=None):
    # we check if a test dataset has been provided, or if at least there is one stored in the tagger
    assert self.test_ds is not None or test_data is not None, "You must provide a test dataset, either to this call or when defining the Tagger"
    # we check if the tagger's model exists and has been trained (that is, history is not empty)
    assert self.model is not None and self.history is not None, "You must first build and train a model, by using build() and train(), before testing"
    
    # we prioritize user-defined datasets, and if they don't exist, we use the one stored in the object
    if test_data is not None:
      results = self.model.evaluate(test_data, batch_size=64)
    elif self.test_ds is not None:
      results = self.model.evaluate(self.test_ds, batch_size=64)
      
    # print the results to the user
    print("test loss, test acc:", results)
  
  # method to obtain the predictions of a given input tensor, both in their "native" form (as in, the output of the softmax layer, a set of measures that
  # can be interpreted as "probability of belonging to the class" for each word) and on a human readable form (a label for each word).
  # keep in mind that the input tensor must be a valid tensorflow tensor.
  def predict(self, input_tensor=None):
    # check that the input tensor has been provided, and that a model exists and has been trained
    assert input_tensor is not None, "You must provide a valid inputs tensor"
    assert self.model is not None and self.history is not None, "You must first build and train a model, by using build() and train(), before making predictions"
  
    # calculate the predictions
    predictions = self.model.predict(input_tensor)
    # create the array of "readable" predictions
    predictions_read = []
    for pred in predictions: # for each native prediction
      readable = []
      for pos in pred: # each native prediction is an array of arrays, each one corresponding to a word
        readable.append(np.argmax(pos)) # we seek the position of the array with the highest value (the highest confidence of belonging to that class) and store it
      predictions_read.append(readable) # once we have the most likely label for each word, we store this as one "readable prediction"
    
    # we return both arrays
    return predictions, predictions_read
  
  # method to perform PoS tagging on a user-defined sentence (although there is a default sentence for completion)
  # it receives the sentence itself, and prints the respective tagging for each word
  def predict_sentence(self, sentence="this is a sample sentence"):
    # we check that a model exists and has been trained
    assert self.model is not None and self.history is not None, "You must first build and train a model, by using build() and train(), before making predictions"
  
    # the same codec we used on the preprocess.py file
    codec={'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12,
         'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17}
    # Convert the sentence to a tensor
    split_sentence=sentence.split()
    size=len(split_sentence)
    sentence=[sentence]
    sentence_tensor = tf.convert_to_tensor(sentence)
    # Make the prediction
    prediction = self.model.predict(sentence_tensor)
    # Calculate the predicted tags for the whole sentence, following a similar method as above
    maxi=np.argmax(prediction,axis=-1)
    for i in range(size): # for each word in the sentence, we print its UPOS tag, and the word itself
        print(self.__key_by_value(codec,maxi[0][i]),'=>',split_sentence[i]) 
  
  # method that saves the built and trained model to a folder
  # it admits the user naming the folder, and defining the save format, although changing that last value is not advisable
  def save(self, folder_name='model', save_format='tf'):
    # we check that a model exists and has been trained (it makes no sense to save the model otherwise)
    assert self.model is not None and self.history is not None, "Build and train the model before trying to save it"
    
    # we save the model
    self.model.save(folder_name, save_format)
    
  # method that loads a trained model from a folder
  # intended to be used in conjunction with the previous method, or at least with the equivalent tensorflow function
  # receives the path to the model folder
  def load(self, folder_path='model'):
    # we load the model to the self.model variable, and we set self.history to True (we do not have the History object anymore, but we know the model is trained
    # and thus checks looking at self.history for that reason should be passed)
    self.model = keras.models.load_model(folder_path)
    self.history = True
 










