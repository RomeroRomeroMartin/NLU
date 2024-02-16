# required imports
import re
import pandas as pd
import numpy as np

# function to preprocess the treebanks; it removes comments, multiwords, and empty nodes.
# receives the path to the treebank file to be processed, and optionally, the name of the output file
def preprocess_dataset(path=None, out="preprocessed_bank.conllu"):
  if path is None: # if we did not get an input file, we can't do anything
    return

  with open(path, 'r', encoding="utf-8") as f: # we open the input file in read mode
    with open(out, 'w', encoding="utf-8") as of: # and the output file in write mode
      for l in f: # for each line on the input file
        if re.search("^#.*", l): # Regular expression to search for comments (lines starting with #)
          continue # if we find a comment, we omit the line
        if re.search("^[0-9]*-[0-9]*\t.*$", l): # Regular expression to search for multiwords (lines where ID is a range)
          continue # omit the line
        if re.search("^[0-9]*\.[0-9]*\t.*$", l): # Regular expression to search for empty nodes (lines where ID is "decimal", i.e., has a dot)
          continue #omit the line
        of.write(l) # if we did not find anything wrong with the line, we write it to the output file
        
# function to encode a UPOS tag as a numerical label (we don't use 0, as it will be used as padding on the samples)
def encode_UPOS(upos_code):
  codec={'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12,
         'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17} # we define a dictionary to encode the UPOS tags
  return codec[upos_code] if upos_code in codec else 17 # and we return the label corresponding to the tag (or 17, for unknown, if it is not on our codec)
  
# function to generate samples from a preprocessed treebank; receives the path to the treebank file
def generate_samples(path=None):
  if path is None: # if we did not get the treebank file path, we can't do anything
    return None

  dataset=pd.DataFrame(columns=['words', 'part-of-speech']) # create the "skeleton" of our samples, a dataframe with a column for the words, and a column for the UPOS labels
  with open(path, 'r', encoding="utf-8") as f: # we open the treebank file in read mode
    words=[] # initialize empty arrays for words and tags
    pos=[]
    for l in f: # reading the file line by line
      if re.search("^\n$", l): # Regular expression that searches for empty lines (they mark a change of sentence in the treebank)
        if len(words)<=128: # we check if the sentence we just finished reading was longer than 128 words or not; and we only proceed if it wasn't
          results=list(map(encode_UPOS, pos)) # we execute the encoding function over all the tags stored in the pos array, and create a list out of the resulting labels
          # we append a new row to our dataframe; the first column is a string, made by joining all the words of the sentence with spaces, and the second column
          # is an array composed by all the labels we encoded before, and padded with zeros until it has exactly 128 elements.
          dataset.loc[len(dataset.index)] = [" ".join(words), np.array(np.pad(results, (0, 128-len(results)), 'constant', constant_values=(0,0)), dtype=np.int32)]
        #we empty the arrays for words and tags, preparing them for the next sentence
        words=[]
        pos=[]
      else: # if we did not find an empty line, we are still on the same sentence
        line_items=l.strip().split('\t') # we strip the line by tabs, and store the word and the tag in their respective arrays.
        words.append(line_items[1])
        pos.append(line_items[3])

  return dataset # once we've processed the full file, we return the dataframe

# this function, given a set of arcs, checks whether or not the sentence is projective
def is_projective(arcs):
  for (i,j) in arcs:
    for (k,l) in arcs:
      if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
        return False #not projective
  return True #is projective

# given a CONLLU file, this generates a Pandas dataframe with all the relevant information for each sentence:
# the "sentence itself" (actually just each word concatenated, separated with spaces), the items of the sentence
# (words and their features), and the set of arcs showcased by the sentence.
def generate_dataframe(path=None):
  if path is None:
    return None

  # generate the dataframe and some variables to store each piece of information
  dataset=pd.DataFrame(columns=['text', 'items', 'arcs'])
  arcs=[]
  items=[]
  sentence=""
  with open(path, 'r', encoding="utf-8") as f: # we open the file in read mode
    for l in f: # reading line by line
      if re.search("^\n$", l): # if we find exclusively a newline character, we finished with the current sentence
        if is_projective([(el[0], el[2]) for el in arcs]): # we check if the sentence is projective, and if it is
          dataset.loc[len(dataset.index)] = [sentence.strip(), items, arcs] # we store its data on the dataframe
        arcs=[] # we clean our variables
        items=[]
        sentence=""
      else: # if we didn't find only a newline, the sentence continues
        if not items: # if items is empty, this is the first word of the sentence, so we have to add ROOT first
          items.append([0, "ROOT", "ROOT", "_", "_", "_", "_", "_", "_", "_"])
        line_items=l.strip().split('\t') # we split the line on tabs
        line_items[0]=int(line_items[0])
        sentence+=line_items[1]+" " # we append the word itself and some whitespace to the sentence
        items.append(line_items) # we append the line items to the item list
        arcs.append((int(line_items[6]),line_items[7],int(line_items[0]))) # we append the arc information to the arcs list
  return dataset