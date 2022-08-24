import numpy as np
import time
import itertools
from State import *


class GameLogic:
  """
  
  """
  def __init__(self,environment,team,game_config,biased_rollout_dict):
    self.game_config = game_config
    self.my_team = team
    self.environment = environment
    self.to_play_order = np.array(('me_to_play','opp_to_play','chance_reward'))
    self.stochastic_role_to_play = 'chance_reward'
    self.batting_actions = [0,1,2,3,4,6]
    self.bowling_actions = ['e','n','a']

    if(self.game_config.is_use_biased_rollout):
      self.biased_rollout_dict = biased_rollout_dict
      # key- key of state, value - list of all nodes having this state
      self.unique_state_keeping_dict = {}

  def fill_unique_statekeeping_dict(self,node):
    node_state = node.state
    key_of_state = node_state.get_state_key_for_dict()

    if(key_of_state in self.unique_state_keeping_dict):
      dict_value = self.unique_state_keeping_dict[key_of_state]
      dict_value.append(node)
      self.unique_state_keeping_dict[key_of_state] = dict_value
    else:
      self.unique_state_keeping_dict[key_of_state] = [node]


  def fill_bias_rollout_dict(self,best_child):
    previous_node = best_child.parent.parent
    previous_state = previous_node.state
    current_best_state = best_child.state
    key_of_state = previous_state.get_state_key_for_dict()
    
    key_of_best_state = current_best_state.get_state_key_for_dict()
    current_best_node_visit_count = 0
    previous_node_visit_count = 0
    for each_best_node in self.unique_state_keeping_dict[key_of_best_state]:
      current_best_node_visit_count += each_best_node.visit_count
      previous_node = each_best_node.parent.parent
      previous_node_visit_count += previous_node.visit_count


    confidence_percent = current_best_node_visit_count / previous_node_visit_count
    
    new_rollout_object = RolloutDictValue(act_bat = current_best_state.batting_action,act_bowl = current_best_state.bowling_action,confidence_percent=confidence_percent)

    if(key_of_state in self.biased_rollout_dict):
      dict_value = self.biased_rollout_dict[key_of_state]
      index_of_current_object = -1
      try:
        index_of_current_object = dict_value.index(new_rollout_object)
      except ValueError:
        pass
      if(index_of_current_object != -1):
        # if(dict_value[index_of_current_object].confidence_percent < confidence_percent):
        self.biased_rollout_dict[key_of_state][index_of_current_object] = new_rollout_object
      else:
        dict_value.append(new_rollout_object)
        self.biased_rollout_dict[key_of_state] = dict_value
    
    else:
      self.biased_rollout_dict[key_of_state] = [new_rollout_object]
    
    # print("$$$$$$$$$$$$$$$$$$$$$$$")
    # print("previous_node: "+str(previous_node))
    # print("best_child: "+str(best_child))
    # print("key_of_state: "+str(key_of_state))
    # print("self.biased_rollout_dict[key_of_state]: ")
    # for each_val in self.biased_rollout_dict[key_of_state]:
    #   print(str(each_val))
    # print("--------------------------")
  

  def get_biased_action(self,key_of_current_state,value_list_of_dict):
    existing_bat_bowl_combinations = []
    prob_of_existing_bat_bowl_combinations = []
    overall_bat_bowl_combinations = list(itertools.product(self.batting_actions,self.bowling_actions))
    sum_confident_percent = 0

    for each_value_of_dict in value_list_of_dict:
      current_batting_action = each_value_of_dict.act_bat
      current_bowling_action = each_value_of_dict.act_bowl
      current_confidence_percent = each_value_of_dict.confidence_percent
      existing_bat_bowl_combinations.append((current_batting_action,current_bowling_action))
      prob_of_existing_bat_bowl_combinations.append(current_confidence_percent)
      sum_confident_percent += current_confidence_percent
    
    # if(np.random.randint(100) == 3): #10% of time print the state to debug
    #   print("key_of_current_state:"+str(key_of_current_state))
    #   print("prob_of_existing_bat_bowl_combinations:"+str(prob_of_existing_bat_bowl_combinations))
    #   print("existing_bat_bowl_combinations:"+str(existing_bat_bowl_combinations))
    #   print("sum_confident_percent:"+str(sum_confident_percent))
    #   for each_val in self.biased_rollout_dict[key_of_current_state]:
    #     print(str(each_val))
    #   print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    
    remaining_bat_bowl_combinations = list(set(overall_bat_bowl_combinations) - set(existing_bat_bowl_combinations))
    prob_of_rem_comb = (1 - min(sum_confident_percent,1))/len(remaining_bat_bowl_combinations)
    
    for each_rem_bat_bowl_comb in remaining_bat_bowl_combinations:
      existing_bat_bowl_combinations.append(each_rem_bat_bowl_comb)
      prob_of_existing_bat_bowl_combinations.append(prob_of_rem_comb)

    try:
      idx = np.random.choice(len(existing_bat_bowl_combinations),1,p=prob_of_existing_bat_bowl_combinations)
    except ValueError:
      print("key_of_current_state:"+str(key_of_current_state))
      print("prob_of_existing_bat_bowl_combinations:"+str(prob_of_existing_bat_bowl_combinations))
      print("existing_bat_bowl_combinations:"+str(existing_bat_bowl_combinations))
      print("sum_confident_percent:"+str(sum_confident_percent))
      for each_val in self.biased_rollout_dict[key_of_current_state]:
        print(str(each_val))
      print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    return_combination = existing_bat_bowl_combinations[idx[0]]

    return return_combination[0],return_combination[1]


  def is_state_terminal(self,given_state):
    if(given_state.num_ball >= self.game_config.horizon_balls):
      return True
    if(given_state.wickets_left == 0):
      return True
    return False

  def get_next_reward_state(self,wicket,runs,previous_state):
    if(previous_state.wickets_left == 0):
      print("Invalid state of batters_list:"+str(previous_state.current_batters_list))
      raise Exception("Invalid state of batters_list:"+str(previous_state.current_batters_list))
    
    nextState = State(total_runs = previous_state.total_runs , feature_batter = previous_state.feature_batter,feature_bowler = previous_state.feature_bowler,ball = (previous_state.num_ball + 1),wickets_left = previous_state.wickets_left,current_batters_list=previous_state.current_batters_list,current_bowlers_list = previous_state.current_bowlers_list)
    if(wicket > 0):
      nextState.wickets_left = nextState.wickets_left - 1
      if(nextState.wickets_left > 0):
        batter_index = self.my_team.get_next_batter(previous_state.current_batters_list)
        previous_batter_index = np.where(previous_state.current_batters_list == -1)[0][0]
        nextState.current_batters_list[previous_batter_index] = 0
        nextState.current_batters_list[batter_index] = -1
        nextState.feature_batter = self.my_team.self_features[batter_index,0:2]

    nextState.total_runs = nextState.total_runs + runs
    
    if ((nextState.num_ball) % 6 == 0 ):
      # print("get_next_reward_state current_bowlers_list:"+str(nextState.current_bowlers_list))
      previous_bowler_index = np.where(nextState.current_bowlers_list < 0)[0][0]
      nextState.current_bowlers_list[previous_bowler_index] = -(nextState.current_bowlers_list[previous_bowler_index] + 1)
      
      ind_array =  np.where(nextState.current_bowlers_list > 0)[0]
      if(ind_array.shape[0] != 0):
        bowler_index = ind_array[0]
        nextState.current_bowlers_list[bowler_index] = -(nextState.current_bowlers_list[bowler_index])
        nextState.feature_bowler = self.my_team.opponent_features[bowler_index,2:4]
    
    
    if(len(nextState.feature_batter.shape)>1):
      print("Invalid shape:"+str(nextState.feature_batter) +" shaaappppeee: "+str(nextState.feature_batter.shape)+"  batter_index:"+str(batter_index)+" previous batters:"+str(previous_state.current_batters_list))
      raise Exception("Invalid shape")

    return nextState


  def get_next_to_play_order(self,current_to_play):
    next_index = np.where(self.to_play_order == current_to_play )[0][0] + 1

    if(next_index == len(self.to_play_order)):
      return self.to_play_order[0]
    return self.to_play_order[next_index]

  def get_possible_actions_in_state(self,to_play,state):
    if(to_play == 'me_to_play'):
      return self.batting_actions
    elif(to_play == 'opp_to_play'):
      return self.bowling_actions

  def get_children_states(self,to_play,state,possible_actions):
    children_states = []
    for each_action in possible_actions:
      child_state = self.get_each_children_state(to_play,state,each_action)
      children_states.append(child_state)
    return children_states
  
  def get_each_children_state(self,to_play,state,each_action):
    if(to_play == 'me_to_play'):
      child_state = State(total_runs= state.total_runs,feature_batter=state.feature_batter,feature_bowler=state.feature_bowler,ball=state.num_ball,wickets_left=state.wickets_left,current_bowlers_list=state.current_bowlers_list,current_batters_list=state.current_batters_list,batting_action=each_action)
      return child_state
    elif(to_play == 'opp_to_play'):
      child_state = State(total_runs = state.total_runs,feature_batter=state.feature_batter,feature_bowler=state.feature_bowler,ball=state.num_ball,wickets_left=state.wickets_left,current_bowlers_list=state.current_bowlers_list,current_batters_list = state.current_batters_list,batting_action=state.batting_action,bowling_action=each_action)
      return child_state
      
  
# If true continue
def within_time_constraints(start_time,time_taken_for_loop,action_timeout):
  current_time =  time. time()
  time_left = (action_timeout - (current_time -start_time))
  if(time_left > max(0.1,time_taken_for_loop)):
    return True
  return False