# required imports
import numpy as np
from oracle import *

# function to extract features from a certain word and state
# it works exactly as it did when preprocessing the samples for training of the models, and those explanations
# can be found on the main notebook, so this will be briefer
def extract_features(items, estado, vectorized_words, n_w, pos_dict):
  # extract a 0-padded array of n_w features from both stack and buffer
  stack_sel=list(np.array(np.pad(estado[0][-n_w:], (n_w-len(estado[0][-n_w:]), 0), 'constant', constant_values=(0, 0)), dtype=np.int32))
  buffer_sel=list(np.array(np.pad(estado[1][0:n_w], (0, n_w-len(estado[1][0:n_w])), 'constant', constant_values=(0, 0)), dtype=np.int32))
  pos_stack = []
  pos_buffer = []
  # extract the PoS-tags for extracted features
  for element in stack_sel:
    if element == 0:
      pos_stack.append(0)
    else:
      pos_stack.append(pos_dict[items[element][3]])
  for element in buffer_sel:
    if element == 0:
      pos_buffer.append(0)
    else:
      pos_buffer.append(pos_dict[items[element][3]])
  # repeat the first operation, but this time using the tokenized form of the words instead of their "sentence_id"
  new_stack=[vectorized_words[indice] for indice in estado[0]]
  new_buffer=[vectorized_words[indice] for indice in estado[1]]
  stack_sel=list(np.array(np.pad(new_stack[-n_w:], (n_w-len(new_stack[-n_w:]), 0), 'constant', constant_values=(0, 0)), dtype=np.int32))
  buffer_sel=list(np.array(np.pad(new_buffer[0:n_w], (0, n_w-len(new_buffer[0:n_w])), 'constant', constant_values=(0, 0)), dtype=np.int32))
  # return everything
  return stack_sel, buffer_sel, pos_stack, pos_buffer

# very similar to the process implemented on the arc-eager oracle, this function takes the set of predictions for action and
# deprel, for a given state, and checks/applies the correct actions.
def check_action_and_state(deprel_pred, action_pred, estado, deprel_dict):
  # we check the predicted deprel
  deprel = np.argmax(deprel_pred)
  # if the network predicted "no deprel", we search the second most likely
  if deprel == 0:
    deprel_pred[deprel] = float('-inf')
    deprel = np.argmax(deprel_pred)
  # we check the predicted action
  action = np.argmax(action_pred)
  # we loop until we find an action that is valid
  while True:
    # if we find that the current action is valid we apply it and break out of the loop
    if action == 1 and la_valid(estado): #left arc
      estado=la_apply_pred(estado, deprel_dict[deprel])
      break
    elif action == 2 and ra_valid(estado): #right arc
      estado=ra_apply_pred(estado, deprel_dict[deprel])
      break
    elif action == 3 : #shift
      estado=shift_apply(estado)
      break
    elif action == 4 and red_valid(estado): #reduce
      estado=red_apply(estado)
      break
    # if the predicted action is not valid, we check the next most probable 
    else:
      action_pred[action] = float('-inf')
      action = np.argmax(action_pred)
  # we return the predicted action and the state after it's been applied
  return action, estado

# repairs a malformed tree
def repair_tree(predicted_arcs, dataframe):
  dataframe_index = predicted_arcs[0] # sentence index on the dataframe
  arcs = predicted_arcs[1] # set of arcs predicted by the model
  terms_with_head = [arc[2] for arc in arcs] # set of words with a head
  terms_without_head = [i for i in range(1, len(dataframe.iloc[dataframe_index, 1])) if i not in terms_with_head] # complementary set
  terms_with_head.sort() # we sort both
  terms_without_head.sort()
  root_arcs = [arc for arc in arcs if arc[1]=='root'] # all arcs classified as root
  root_arcs.sort(key=lambda x: x[2]) # sorted ascendingly by the target of the arc

  # add root node if there isn't one
  # if there isn't, add leftmost word without head (that is, lowest id not present on the right side of an arc) as root.
  if len(root_arcs) < 1:
    if len(terms_without_head) == 0:
      print("No free words to classify as root, and root missing")
    else:
      arcs.append((0, 'root', terms_without_head[0]))
      terms_without_head.pop(0)

  # get single root if there are multiple ones
  if len(root_arcs) > 1:
    for i in range(1, len(root_arcs)):
      arcs.remove(root_arcs[i])
      arcs.append((root_arcs[0][2], '_', root_arcs[i][2]))

  # if there are terms with no head, hang them from the "dependant" of root
  root_dependant = [arc[2] for arc in arcs if arc[1]=='root'][0]
  for term in terms_without_head:
    arcs.append((root_dependant, '_', term))