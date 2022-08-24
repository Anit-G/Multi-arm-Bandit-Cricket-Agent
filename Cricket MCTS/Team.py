import numpy as np
from State import *
from MCTSAlgo import *

class Team:
  def __init__(self,self_features,opponent_features,environment):
    self.self_features = self_features
    self.opponent_features = opponent_features
    self.explore_wicket = 0
    self.explore_runs = 0
    self.environment = environment
    self.current_batters_list =np.array([1,1,1,1,1])  # 0 indicates player who had got out, -1 indicates currently playing bowler in current state, 1 indicates bowler yet to bat
    self.current_bowlers_list =np.array([2,2,2,2,2]) # -number indicates currently playing bowler, +ve number indicates number of overs left

    self.batting_order = [0,1,2,3,4]
    self.biased_rollout_dict = {}

  # Returns index of player who is not-out as per batting order
  def get_next_batter(self,current_batters_list = None):
    if(current_batters_list is None):
      current_batters_list = self.current_batters_list
    
    for each_order in self.batting_order:
      # If batter is currently batting or already out, skip him
      if(current_batters_list[each_order] > 0):
        return each_order
    
  # For phase 2
  # def get_next_bowler(self):
  #   return np.random.randint(0,5)

  def get_batting_action(self,ball,total_runs,wickets_left,score_to_chase,feature_batter,feature_bowler,current_batter_index,current_bowler_index,action_timeout,iterative_deepening_depth=6,ucb_mode=None,num_rand_rollout_multiplier=None):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Ball ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   "+str(ball))
    # batter = -1
    # for index_of_each_full_feature in range(len(self.self_features)):
    #   if((self.self_features[index_of_each_full_feature,0:2] == feature_batter).all()):
    #     batter = index_of_each_full_feature
    ind_array = np.where(self.current_batters_list == -1)[0]
    if(ind_array.shape[0] != 0):
      previous_batter_index = ind_array[0]
      self.current_batters_list[previous_batter_index] = 0
    self.current_batters_list[current_batter_index] = -1

    # bowler = -1
    previous_bowler_index = -1
    # for index_of_each_full_feature in range(len(self.opponent_features)):
    #   if((self.opponent_features[index_of_each_full_feature,2:4] == feature_bowler).all()):
    #     bowler = index_of_each_full_feature
    ind_array = np.where(self.current_bowlers_list < 0)[0]
    if(ball != 0):
      previous_bowler_index = ind_array[0]
      if(ball % 6 == 0):
        self.current_bowlers_list[previous_bowler_index] = (self.current_bowlers_list[previous_bowler_index] + 1)
    
    if(previous_bowler_index == -1 or current_bowler_index != previous_bowler_index):
      self.current_bowlers_list[current_bowler_index] = -(self.current_bowlers_list[current_bowler_index])

    game_config = GameConfig(60,iterative_deepening_depth=iterative_deepening_depth,ucb_mode=ucb_mode,num_rand_rollout_multiplier=num_rand_rollout_multiplier,is_use_biased_rollout=False)
    print("get_batting_action Current batters list:",self.current_batters_list)
    print("get_batting_action Current bowlers list:",self.current_bowlers_list)
    mctsAlgo = MCTSAlgo(environment = self.environment,team = self,game_config = game_config)
    current_state = State(ball = ball,total_runs=0,wickets_left = wickets_left,feature_batter=feature_batter,feature_bowler=feature_bowler,current_batters_list=self.current_batters_list,current_bowlers_list=self.current_bowlers_list,batting_action=None,bowling_action=None)
    # batting_action,biased_rollout_dict = mctsAlgo.run(current_state,action_timeout,my_role='batting',biased_rollout_dict=self.biased_rollout_dict)
    batting_action,biased_rollout_dict = mctsAlgo.run(current_state,action_timeout,my_role='batting',biased_rollout_dict={})
    self.biased_rollout_dict = biased_rollout_dict
    
    print("___________Batting action________________: "+str(batting_action))
    
    return batting_action

  def get_bowling_action(self,ball,total_runs,wickets_left,score_to_chase,feature_batter,feature_bowler,current_batter_index,current_bowler_index):
    bowling_action = np.random.randint(1,4) # this is the place to code UCTS
    return bowling_action
  
  def get_random_batting_action(self,ball,total_runs,wickets_left,score_to_chase,feature_batter,feature_bowler,action_timeout):
    print("~~~~~~~Ball ~~~~~~~~~"+str(ball))
    batting_action = np.random.randint(0,6)  # this is the place to code UCTS
    return batting_action

  def explore(self,explore_num_balls=20):
      """
          Args:---------------------------
          explore_num_balls: int
                            Number of balls provided for testing the capability of 
                            opponent team.
                            (number of balls allower to decide batting order)
          Return:------------------------
          batting order
      """
      balls_per_player = int(explore_num_balls/10)
      batting_features = self.self_features[:,0:2]
      #Run the two scenarios
      results = []
      for case,act_bat,act_bowl,feature_bowler in zip(['Best','Worst'],[6,0],[3,1],np.array([[1,1],[5,5]])):
        result = []
        for j,feature_batter in enumerate(batting_features):
          print(f'Batting: Player {j}')
          wicket = 0
          runs = 0
          for i in range(balls_per_player):
            #print(f'Ball Number: {i+1}')
            w,r = self.environment.get_outcome(feature_batter,feature_bowler,act_bat,act_bowl)
            wickets = wicket+w
            runs =  runs+r
          print(f"Total runs: {runs}")
          result.append(runs) # shape = (5,1)
        results.append(result) # shape = (2,5,1) = (scenario,number of players,runs)
      results = np.array(results)
      # find median scores
      median_scores = np.median(results,0)

      # greedily arrange batting order based on results
      batting_order = np.argsort(median_scores)[::-1]
      self.batting_order = batting_order
      print("Batting order is:"+str(batting_order))

      return batting_order

class Australia(Team):
  def __init__(self,self_features,opponent_features,environment):
    super().__init__(self_features,opponent_features,environment)

class India(Team):
  def __init__(self,self_features,opponent_features,environment):
    super().__init__(self_features,opponent_features,environment)
