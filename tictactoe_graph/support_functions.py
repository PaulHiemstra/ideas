def add_options_to_node(tree, node, tt_data, player, remaining_options):
    for option in remaining_options:
        local_tt_data = copy.deepcopy(tt_data)           # To prevent changing these values in other branches of the tree
        local_tt_data.make_move(player, option, False)
        if node.identifier != 'root':
            new_identifier = node.identifier + option
        else:
            new_identifier = option
        tree.create_node(option, new_identifier, node.identifier, data = local_tt_data)
        if len(remaining_options) > 1 and not local_tt_data.is_endstate():
            add_options_to_node(tree, tree[new_identifier], local_tt_data, 
                                flip_player[player], remove_value_list(remaining_options, option))
    return None

class Memoize_tree:
    '''
    From https://www.python-course.eu/python3_memoization.php
    '''
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        function_call_hash = args[1:]  # Note we skip the first argument, this is the tree that is always the same. Adding this would slow down the hashing procedure
        if function_call_hash not in self.memo:
            self.memo[function_call_hash] = self.fn(*args)
        return self.memo[function_call_hash]

@Memoize_tree
def minmax_tt(tree, current_id, is_max):
    #print('Dealing with id: ', current_id)
    current_node = tree[current_id] 
    if current_node.data.is_endstate():
        return current_node.data.get_value()
    children_of_current_id = tree.children(current_id)
    scores = [minmax_tt(tree, child.identifier, not is_max) for child in children_of_current_id]
    if is_max:
        return max(scores)
    else:
        return min(scores)

def determine_move(tree, current_id, is_max):
    potential_moves = tree.children(current_id)
    moves = [child.identifier[-1] for child in potential_moves]
    raw_scores = [minmax_tt(tree, child.identifier, not is_max) for child in potential_moves]
    #print(dict(zip(moves, raw_scores)))
    if is_max:
        return moves[raw_scores.index(max(raw_scores))]
    else:
        return moves[raw_scores.index(min(raw_scores))]
    
import random
from tqdm import tqdm

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value
         
         
     Based on https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary"""  
     k=list(d.keys())
     # boltzmann
     v = np.array(list(d.values()))
     return k[int(random.choice(np.argwhere(v == np.amax(v))))]  # If there are multiple max values, choose randomly

class Player:
    def __init__(self, id, alpha = 0.5, gamma = 0.6, epsilon = 0.1):
        self.qtable = {}
        self.id = id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def get_qtable(self):
        return self.qtable
    def get_id(self):
        return self.id
    def set_params(self, 
                   alpha = 0.5,       # How fast do we learn from new info
                   gamma = 0.6,       # How much are we focused on the short or the long term. 1 = max long term, 0 is max short term
                   epsilon = 0.1):    # exploration vs exploitation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def make_move(self, game, opponent_qtable, verbose=False):
        if game.is_endstate():
            # If the game is done by the time we get to make a move, simply skip this step
            return game
        # Make a choice what move to take next
        possible_moves = game.get_possible_next_moves()
        current_state = game.get_current_state()
        
        # If the current_state does not exist in the qtable, insert it
        if current_state not in self.qtable:
            # New entry in the qtable, init to zero. 
            #self.qtable[current_state] = dict(zip([current_state + move for move in possible_moves], 
            #                                      np.random.uniform(0, 0.1, len(possible_moves))))  # initialize on a small amount of random noise. Promotes varyiety
            self.qtable[current_state] = dict(zip(possible_moves, 
                                                  np.zeros(len(possible_moves))))  
            
        # Insert epsilon choice here, exploit or explore
        if random.uniform(0, 1) < self.epsilon:
            new_state, reward, action = game.make_move(self.id, random.choice(possible_moves))   # Random choice
        else:  # Exploit our qtable
            new_state, reward, action = game.make_move(self.id, keywithmaxval(self.qtable[current_state]))   # Optimal choice
        if self.epsilon == 0:    # If we set epsilon to 0, we only want to play. No updating needed. 
            return game
        
        # Update the qtable
        # Example qtable entry
        #  
        #  qtable['abd'] = {'abde': 2.1, 'abdf': 1.3, etc}
        # 
        # where 'abde': 2.1 is the q-value of taking action 'e' in state 'abd'. 
        # Note that the qtable is ragged. Not all moves are possible from each state
        old_value = self.qtable[current_state][action]
        try:
            next_max = max(-np.array(list(opponent_qtable[new_state].values())))
        except KeyError:  # In case the tree for next state has not been made yet, simply return 0
            next_max = 0
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        if verbose:
            print('old: ', old_value, 'new_value: ', new_value, 'alpha: ', self.alpha, 'reward: ', reward, 'gamma: ', self.gamma, 'next_max: ', next_max)
        self.qtable[current_state][action] = new_value

        if verbose:
            print(self.id, current_state, self.qtable[current_state], new_state, action, old_value, new_value)
        
        return game

class Player_tree:
    def __init__(self, tree, id):
        self.tree = tree
        self.id = id
    def make_move(self, moves_made):
        if self.id == -1:
            return determine_move(self.tree, moves_made, False)
        else:
            return determine_move(self.tree, moves_made, True)

def play_tictactoe(no_episodes, player1, player2, verbose=False):
    tactoe = Tictoe(3)       

    p1_rewards = np.zeros(no_episodes)
    p2_rewards = np.zeros(no_episodes)
    paths = np.chararray(no_episodes, itemsize=9)
    for ep_idx in tqdm(range(no_episodes)):
        stop = False
        while not stop:
            tactoe = player1.make_move(tactoe, player2.get_qtable(), verbose=verbose)
            tactoe = player2.make_move(tactoe, player1.get_qtable(), verbose=verbose)
            if tactoe.is_endstate():
                p1_rewards[ep_idx] = tactoe.get_reward(player1.get_id())
                p2_rewards[ep_idx] = tactoe.get_reward(player2.get_id())
                paths[ep_idx] = tactoe.get_moves_made()
                tactoe.reset_board()
                stop = True
    return [player1, player2, p1_rewards, p2_rewards, paths]

import random
from tqdm import tqdm

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value
         
         
     Based on https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary"""  
     k=list(d.keys())
     # boltzmann
     v = np.array(list(d.values()))
     return k[int(random.choice(np.argwhere(v == np.amax(v))))]  # If there are multiple max values, choose randomly

class Player:
    def __init__(self, id, alpha = 0.5, gamma = 0.6, epsilon = 0.1):
        self.qtable = {}
        self.id = id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def get_qtable(self):
        return self.qtable
    def get_id(self):
        return self.id
    def set_params(self, 
                   alpha = 0.5,       # How fast do we learn from new info
                   gamma = 0.6,       # How much are we focused on the short or the long term. 1 = max long term, 0 is max short term
                   epsilon = 0.1):    # exploration vs exploitation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def make_move(self, game, opponent_qtable, verbose=False):
        if game.is_endstate():
            # If the game is done by the time we get to make a move, simply skip this step
            return game
        # Make a choice what move to take next
        possible_moves = game.get_possible_next_moves()
        current_state = game.get_current_state()
        
        # If the current_state does not exist in the qtable, insert it
        if current_state not in self.qtable:
            # New entry in the qtable, init to zero. 
            #self.qtable[current_state] = dict(zip([current_state + move for move in possible_moves], 
            #                                      np.random.uniform(0, 0.1, len(possible_moves))))  # initialize on a small amount of random noise. Promotes varyiety
            self.qtable[current_state] = dict(zip(possible_moves, 
                                                  np.zeros(len(possible_moves))))  
            
        # Insert epsilon choice here, exploit or explore
        if random.uniform(0, 1) < self.epsilon:
            new_state, reward, action = game.make_move(self.id, random.choice(possible_moves))   # Random choice
        else:  # Exploit our qtable
            new_state, reward, action = game.make_move(self.id, keywithmaxval(self.qtable[current_state]))   # Optimal choice
        if self.epsilon == 0:    # If we set epsilon to 0, we only want to play. No updating needed. 
            return game
        
        # Update the qtable
        # Example qtable entry
        #  
        #  qtable['abd'] = {'abde': 2.1, 'abdf': 1.3, etc}
        # 
        # where 'abde': 2.1 is the q-value of taking action 'e' in state 'abd'. 
        # Note that the qtable is ragged. Not all moves are possible from each state
        old_value = self.qtable[current_state][action]
        try:
            next_max = max(-np.array(list(opponent_qtable[new_state].values())))
        except KeyError:  # In case the tree for next state has not been made yet, simply return 0
            next_max = 0
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        if verbose:
            print('old: ', old_value, 'new_value: ', new_value, 'alpha: ', self.alpha, 'reward: ', reward, 'gamma: ', self.gamma, 'next_max: ', next_max)
        self.qtable[current_state][action] = new_value

        if verbose:
            print(self.id, current_state, self.qtable[current_state], new_state, action, old_value, new_value)
        
        return game

def play_tictactoe(no_episodes, player1, player2, verbose=False):
    tactoe = Tictoe(3)       

    p1_rewards = np.zeros(no_episodes)
    p2_rewards = np.zeros(no_episodes)
    paths = np.chararray(no_episodes, itemsize=9)
    for ep_idx in tqdm(range(no_episodes)):
        stop = False
        while not stop:
            tactoe = player1.make_move(tactoe, player2.get_qtable(), verbose=verbose)
            tactoe = player2.make_move(tactoe, player1.get_qtable(), verbose=verbose)
            if tactoe.is_endstate():
                p1_rewards[ep_idx] = tactoe.get_reward(player1.get_id())
                p2_rewards[ep_idx] = tactoe.get_reward(player2.get_id())
                paths[ep_idx] = tactoe.get_moves_made()
                tactoe.reset_board()
                stop = True
    return [player1, player2, p1_rewards, p2_rewards, paths]