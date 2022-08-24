import math
from operator import attrgetter
import random

class MCTSNode:
  def __init__(self,parent,to_play,state,isNodeStochastic = False):
    self.visit_count = 0
    self.to_play = to_play
    # Dictionary of action,node array (bcoz sometimes, same action can lead to more than one node in case of stochastic actions)
    self.children = []
    self.state = state
    self.isNodeStochastic = False
    self.parent = parent
    # When a state is stochastic
    self.nodeOccurenceProbability = 0
    self.valueOfNode = 0
    self.ucb_scoreOfNode = 0

  def expand_children(self,gameLogic):
     
    children_nodes = []
    possible_actions = gameLogic.get_possible_actions_in_state(self.to_play,self.state)
    children_states = gameLogic.get_children_states(self.to_play,self.state,possible_actions)
    for each_child_state in children_states:
      next_to_play = gameLogic.get_next_to_play_order(self.to_play)
      child_node = MCTSNode(parent=self,to_play=next_to_play,state=each_child_state)
      if(next_to_play == gameLogic.stochastic_role_to_play):
        child_node.isNodeStochastic = True
        if(gameLogic.game_config.is_use_biased_rollout):
          gameLogic.fill_unique_statekeeping_dict(child_node)
      children_nodes.append(child_node)
    self.children = children_nodes
  
  def expand_stochastic_nodes(self,gameLogic):
    # print("$$$$$$$$$$$$$$$$$$$ Stochastic expand $$$$$$$$$$$$$$$$$$$$$$")
    wicket,runs = gameLogic.environment.get_outcome(self.state.feature_batter, self.state.feature_bowler, self.state.batting_action, self.state.bowling_action)
    new_state = gameLogic.get_next_reward_state(wicket,runs,self.state)
    for each_child in self.children:
      # print("Each child state: "+str(each_child.state.total_runs)+" \n New state: "+str(new_state.total_runs))
      if(each_child.state == new_state):
        each_child.visit_count += 1
        each_child.parent.update_node_probability()
        return each_child
    
    next_to_play = gameLogic.get_next_to_play_order(self.to_play)
    child_node = MCTSNode(parent=self,to_play=next_to_play,state=new_state)
    child_node.visit_count = 1

    child_node.parent.update_node_probability()
    
    self.children.append(child_node)

    return child_node
  
  def update_node_probability(self):
    for each_child in self.children:
      each_child.nodeOccurenceProbability = each_child.visit_count / each_child.parent.visit_count

  def select_best_child(self,gameLogic,round_num=None):
    if(round_num is None):
      round_number = self.parent.visit_count
    else:
      round_number = round_num

    for i in range(len(self.children)):
      if(self.children[i].visit_count == 0):
        self.children[i].visit_count = 1
        if(self.children[i].isNodeStochastic):
          return self.children[i].expand_stochastic_nodes(gameLogic)
        return self.children[i]
    for each_child_node in self.children:
      each_child_node.compute_ucb_score(round_number,gameLogic)
    best_child =  max(self.children , key = attrgetter('ucb_scoreOfNode'))
    best_child.visit_count = best_child.visit_count + 1
    if(best_child.isNodeStochastic):
      if(gameLogic.game_config.is_use_biased_rollout):
          gameLogic.fill_bias_rollout_dict(best_child)
      return best_child.expand_stochastic_nodes(gameLogic)
    return best_child


  def compute_ucb_score(self,round_number,gameLogic):
    # All the modes are just for comparision currently. Only best one will be fixed later
    if(gameLogic.game_config.ucb_mode is None):
      # Tighter bound version of UCB
      exploration_term = math.sqrt(2 * math.log(1 + round_number * math.pow(math.log(round_number),2)) / self.visit_count)
    elif(gameLogic.game_config.ucb_mode == 'VERYLOOSE'):
      # Another very loose bound version for UCB
      exploration_term = math.sqrt(4 * math.log(round_number) / self.visit_count)  
    elif(gameLogic.game_config.ucb_mode == 'LOOSE'):
      # Another looser bound version for UCB
      exploration_term = math.sqrt(2 * math.log(round_number) / self.visit_count)  
    elif(gameLogic.game_config.ucb_mode == 'GREEDY'):
      # Greedy choice means no UCB concept used at all
      exploration_term = 0

    if(self.to_play == 'me_to_play'):
      self.ucb_scoreOfNode = self.valueOfNode + exploration_term
    elif(self.to_play == 'opp_to_play'):
      self.ucb_scoreOfNode = self.valueOfNode + exploration_term
  
  def simulate_random_rollout(self,gameLogic,non_random_mode=None):
    
    if(gameLogic.is_state_terminal(self.state)):
      return self.state.wickets_left,self.state.total_runs
    
    if(len(self.state.feature_batter.shape)>1):
      print("Simulation Invalid shape:"+str(self.state.feature_batter) +" shaaappppeee: "+str(self.state.feature_batter.shape)+" previous batters:"+str(self.state.current_batters_list))
      raise Exception("Simulation Invalid shape")
    
    if(self.to_play != 'chance_reward'):
      next_to_play = gameLogic.get_next_to_play_order(self.to_play)
      possible_actions = gameLogic.get_possible_actions_in_state(self.to_play,self.state)
      random_action = random.choice(possible_actions)
      next_child_state = gameLogic.get_each_children_state(self.to_play,self.state,random_action)
      if(gameLogic.is_state_terminal(next_child_state)):
        return next_child_state.wickets_left,next_child_state.total_runs
      next_child_node = MCTSNode(parent=self,to_play=next_to_play,state=next_child_state)
      return next_child_node.simulate_random_rollout(gameLogic)


    number_of_sampling_per_call = 5
    sum_of_runs = 0
    sum_of_wickets = 0
    if(non_random_mode is None):
      if(not(gameLogic.game_config.is_use_biased_rollout)):
          chosen_batting_action = random.choice(gameLogic.batting_actions)
          chosen_bowling_action = random.choice(gameLogic.bowling_actions)
      else:
        dict_rollout = gameLogic.biased_rollout_dict
        key_of_current_state = self.state.get_state_key_for_dict()
        if(key_of_current_state in dict_rollout):
          value_of_dict = dict_rollout[key_of_current_state]
          chosen_batting_action,chosen_bowling_action = gameLogic.get_biased_action(key_of_current_state,value_of_dict)
        else:
          chosen_batting_action = random.choice(gameLogic.batting_actions)
          chosen_bowling_action = random.choice(gameLogic.bowling_actions)
    else:
      if(non_random_mode == 'best'):
        chosen_batting_action = 6
        chosen_bowling_action = 'a'
      elif(non_random_mode == 'worst'):
        chosen_batting_action = 0
        chosen_bowling_action = 'e'
      elif(non_random_mode == 'med'):
        chosen_batting_action = 3
        chosen_bowling_action = 'n'
      else:
        if(not(gameLogic.game_config.is_use_biased_rollout)):
          chosen_batting_action = random.choice(gameLogic.batting_actions)
          chosen_bowling_action = random.choice(gameLogic.bowling_actions)
        else:
          dict_rollout = gameLogic.biased_rollout_dict
          key_of_current_state = self.state.get_state_key_for_dict()
          if(key_of_current_state in dict_rollout):
            value_of_dict = dict_rollout[key_of_current_state]
            chosen_batting_action,chosen_bowling_action = gameLogic.get_biased_action(key_of_current_state,value_of_dict)
          else:
            chosen_batting_action = random.choice(gameLogic.batting_actions)
            chosen_bowling_action = random.choice(gameLogic.bowling_actions)
          

    for i in range(number_of_sampling_per_call):
      wicket,run = gameLogic.environment.get_outcome(self.state.feature_batter, self.state.feature_bowler, chosen_batting_action, chosen_bowling_action)
      sum_of_wickets+= wicket
      sum_of_runs += run
    average_runs = sum_of_runs / number_of_sampling_per_call
    average_wicket = sum_of_wickets / number_of_sampling_per_call
    if(average_wicket > 0.5):
      wicket = 1
    else:
      wicket = 0
    
    new_state = gameLogic.get_next_reward_state(wicket,average_runs,self.state)
    
    if(gameLogic.is_state_terminal(new_state)):
      return new_state.wickets_left,new_state.total_runs
    next_child_node = MCTSNode(parent=self,to_play=self.to_play,state=new_state)
    return next_child_node.simulate_random_rollout(gameLogic)
  
  def back_propogate_node_value(self,gameLogic,v_list=None):
    if(v_list is None):
      value_list = []
    else:
      value_list = v_list

    if(self.parent is None):
      return value_list
    
    if(self.parent.isNodeStochastic):
      all_children = self.parent.children
      expected_value = 0
      prob_sum_for_sanity_check = 0
      for each_children in all_children:
        prob_sum_for_sanity_check += each_children.nodeOccurenceProbability
        expected_value += (each_children.nodeOccurenceProbability * each_children.valueOfNode)
      
      # Purely sanity check code to catch issues. Check is whether sum of probs = 1
      if((prob_sum_for_sanity_check - 1) > 0.001):
        print("sum of probs: "+str(prob_sum_for_sanity_check))
        print("nodeOccurenceProbability doesn't sum to 1 for parent object: "+str(self.parent))
        raise Exception("nodeOccurenceProbability doesn't sum to 1")
      
      self.parent.update_value_of_node(expected_value)
      value_list.append(expected_value)
    else:
      self.parent.update_value_of_node(self.valueOfNode)
      value_list.append(self.valueOfNode)

    return self.parent.back_propogate_node_value(gameLogic,value_list)
    
      
  # Phase 2 needs shift(maybe)
  def update_value_of_node(self,value):
    if(math.isinf(value)):
      print("Error at object: "+str(self))
      raise Exception("Value is infinity. Bug exists")
    # Value is always positive because we want to maximize value (hence runs scored too)
    if(self.to_play == 'me_to_play' or self.to_play == 'chance_reward'):
      if(value > 0):
        self.valueOfNode = self.update_mean(self.valueOfNode,-value,self.visit_count) 
      else:
        self.valueOfNode = self.update_mean(self.valueOfNode,value,self.visit_count) 
    # Value is always negative because we want to maximize value (minimize runs scored)
    elif(self.to_play == 'opp_to_play'):
      if(value < 0):
        self.valueOfNode = self.update_mean(self.valueOfNode,-value,self.visit_count) 
      else:
        self.valueOfNode = self.update_mean(self.valueOfNode,value,self.visit_count) 

  def update_mean(self,current_mean, new_value, n ):
    """ calculate the new mean from the previous mean and the new value """
    ret = (1 - 1.0/n) * current_mean + (1.0/n) * new_value
    if(math.isinf(ret)):
      print("Error. current_mean:"+str(current_mean)+" new_value:"+str(new_value)+" n:"+str(n))
      raise Exception("Value is infinity. Bug exists")
    return ret

  def is_node_expanded(self):
    if(len(self.children)==0):
      return False
    return True
  
  def __eq__(self, other) : 
    return self.__dict__ == other.__dict__

  def __str__(self):
    return "\n Visit_count: "+str(self.visit_count)+" to_play:"+str(self.to_play)+" isNodeStochastic: "+str(self.isNodeStochastic)+" valueOfNode:"+str(self.valueOfNode)+" nodeOccurenceProbability: "+str(self.nodeOccurenceProbability)+" ucb_scoreOfNode:"+str(self.ucb_scoreOfNode)+" \n state:"+str(self.state)
    