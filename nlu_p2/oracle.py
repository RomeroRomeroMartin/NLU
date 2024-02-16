# creates the initial state for a sentence, based on its items (words and their features)
def initial_state(items):
  stack=[]
  buffer=[]
  action=None
  arcs=[]

  # the stack initially contains only ROOT (the first item), while the buffer holds all other words
  stack.append(items[0][0])
  for item in items[1:]:
    buffer.append(item[0])

  return [stack, buffer, action, arcs]

# checks if a state is final
def final_state(state):
  if not state[1]: # if the buffer is empty, we're done
    return True
  else:
    return False

# checks if a left arc action is valid for the current state
def la_valid(state):
  stack = state[0]
  arcs = state[3]

  # invalid if the head of the stack is root or if there's already an arc pointing to it
  if stack[-1]==0 or [tup for tup in arcs if tup[2]==stack[-1]]:
    return False
  else:
    return True

# checks if a left arc action is correct for the current state, based on the ground truth
def la_correct(state, truth):
  stack = state[0]
  buffer = state[1]
  # if the arc to be constructed exist on the ground truth, then it is correct
  if [tup for tup in truth if tup[0]==buffer[0] and tup[2]==stack[-1]]:
    return True
  else:
    return False

# applies a left arc action, giving it the same dependency relation it has on the ground truth
def la_apply(state, truth):
  stack = state[0]
  buffer = state[1]
  action = 'left-arc'
  arcs = state[3]

  arcs.append([tup for tup in truth if tup[0]==buffer[0] and tup[2]==stack[-1]][0])
  stack.pop()

  return [stack, buffer, action, arcs]

# applies a left arc action, giving it an arbitrary dependency relation
def la_apply_pred(state, deprel):
  stack = state[0]
  buffer = state[1]
  action = 'left-arc'
  arcs = state[3]

  arcs.append((buffer[0], deprel, stack[-1]))
  stack.pop()

  return [stack, buffer, action, arcs]

# checks if a right arc action is valid for the current state
def ra_valid(state):
  buffer = state[1]
  arcs = state[3]

  # invalid if the head of the buffer already has an arc pointing at it
  if [tup for tup in arcs if tup[2]==buffer[0]]:
    return False
  else:
    return True

# checks if a right arc action is correct for the current state, based on the ground truth
def ra_correct(state, truth):
  stack = state[0]
  buffer = state[1]
  # if the arc to be built exists on the ground truth, then it is correct
  if [tup for tup in truth if tup[0]==stack[-1] and tup[2]==buffer[0]]:
    return True
  else:
    return False

# applies a right arc action, giving it the same dependency relation it has on the ground truth
def ra_apply(state, truth):
  stack = state[0]
  buffer = state[1]
  action = 'right-arc'
  arcs = state[3]

  arcs.append([tup for tup in truth if tup[0]==stack[-1] and tup[2]==buffer[0]][0])
  stack.append(buffer[0])
  buffer.pop(0)

  return [stack, buffer, action, arcs]

# applies a right arc action, giving it an arbitrary dependency relation
def ra_apply_pred(state, deprel):
  stack = state[0]
  buffer = state[1]
  action = 'right-arc'
  arcs = state[3]

  arcs.append((stack[-1], deprel, buffer[0]))
  stack.append(buffer[0])
  buffer.pop(0)

  return [stack, buffer, action, arcs]

# checks if a reduce action is valid for the current state
def red_valid(state):
  stack = state[0]
  arcs = state[3]

  # invalid if there are no arcs that point to the head of the stack
  if not [tup for tup in arcs if tup[2]==stack[-1]]:
    return False
  else:
    return True

# checks if a reduce action is correct for the current state, based on ground truth
def red_correct(state, truth):
  stack = state[0]
  arcs = state[3]

  # we compute the arcs involving the head of the stack (in any position) for the groun truth and our current set of arcs
  real_head_arcs = [(tup[0], tup[2]) for tup in truth if tup[0]==stack[-1] or tup[2]==stack[-1]]
  current_head_arcs = [(tup[0], tup[2]) for tup in arcs if tup[0]==stack[-1] or tup[2]==stack[-1]]

  # if the sets are equivalent, then the action is correct
  if set(real_head_arcs) == set(current_head_arcs):
    return True
  else:
    return False

# applies a reduce action to the current state
def red_apply(state):
  stack = state[0]
  buffer = state[1]
  action = 'reduce'
  arcs = state[3]

  stack.pop()

  return [stack, buffer, action, arcs]

# applies a shift action to the current state
def shift_apply(state):
  stack = state[0]
  buffer = state[1]
  action = 'shift'
  arcs = state[3]

  stack.append(buffer[0])
  buffer.pop(0)

  return [stack, buffer, action, arcs]

# implements an arc-eager oracle for sentences, using the items of the sentence and the set of arcs as the gold standard
def oracle(items, arcs):
  states=[] # list to store all states in the execution trace
  s=initial_state(items) # generate the starting state for the sentence

  while not final_state(s): # while my current state is not final
    # check if the action is valid and correct, and if it is, apply it and register it on the trace
    # this is tried for all actions
    if la_valid(s) and la_correct(s, arcs): 
      action = 'left-arc'
      arc = [tup for tup in arcs if tup[0]==s[1][0] and tup[2]==s[0][-1]][0]
      states.append([s[0][:], s[1][:], action, s[3][:]+[arc]])
      s=la_apply(s, arcs)
    elif ra_valid(s) and ra_correct(s, arcs):
      action = 'right-arc'
      arc = [tup for tup in arcs if tup[0]==s[0][-1] and tup[2]==s[1][0]][0]
      states.append([s[0][:], s[1][:], action, s[3][:]+[arc]])
      s=ra_apply(s, arcs)
    elif red_valid(s) and red_correct(s, arcs):
      action = 'reduce'
      states.append([s[0][:], s[1][:], action, s[3][:]])
      s=red_apply(s)
    else:
      action = 'shift'
      states.append([s[0][:], s[1][:], action, s[3][:]])
      s=shift_apply(s)
  states.append([s[0][:], s[1][:], 'end', s[3][:]]) # append a last state to the trace

  return states # return the trace

