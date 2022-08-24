import numpy as np

class State:
  def __init__(self,ball,total_runs,wickets_left,feature_batter,feature_bowler,current_batters_list,current_bowlers_list,batting_action=None,bowling_action=None):
    self.feature_batter = np.copy(feature_batter)
    self.feature_bowler = np.copy(feature_bowler)
    self.num_ball = ball
    self.total_runs = total_runs
    self.wickets_left = wickets_left
    self.bowling_action = bowling_action
    self.batting_action = batting_action
    self.current_bowlers_list = np.copy(current_bowlers_list)
    self.current_batters_list = np.copy(current_batters_list)
  
  def get_state_key_for_dict(self):
    return "fb"+str(self.feature_batter)+"fo"+str(self.feature_bowler)+"b"+str(self.num_ball)+"w"+str(self.wickets_left)+"t"+str(self.total_runs)
  
  def __eq__(self, other) : 
    # print("self: "+str(self))
    # print("other: "+str(other))
    check1 = np.array_equal(self.feature_batter,other.feature_batter)
    # print("check1:",check1)
    check2 = np.array_equal(self.feature_bowler,other.feature_bowler)
    # print("check2:",check2)
    check3 = (self.num_ball == other.num_ball)
    # print("check3:",check3)
    check4 = (self.total_runs == other.total_runs)
    # print("check4:",check4)
    check5 = (self.wickets_left == other.wickets_left)
    # print("check5:",check5)
    check6 = (self.bowling_action == other.bowling_action)
    # print("check6:",check6)
    check7 = (self.batting_action == other.batting_action)
    # print("check7:",check7)
    check8 = np.array_equal(self.current_bowlers_list,other.current_bowlers_list)
    # print("check8:",check8)
    check9 = np.array_equal(self.current_batters_list,other.current_batters_list)
    # print("check9:",check9)

    return ( check1 and check2 and check3 and check4 and check5 and check6 and check7 and check8 and check9)
  
  def __str__(self):
    return "feature_batter: "+str(self.feature_batter) + "\n feature_bowler: "+str(self.feature_bowler)+"\n num_ball: "+str(self.num_ball)+" total_runs: "+str(self.total_runs)+" wickets_left: "+str(self.wickets_left)+"\n bowling_action: "+str(self.bowling_action)+"\n batting_action: "+str(self.batting_action)+" \n current_bowlers_list: "+str(self.current_bowlers_list)+"\n current_batters_list: "+str(self.current_batters_list)


class RolloutDictValue:
  def __init__(self,act_bat,act_bowl,confidence_percent):
    self.act_bat = act_bat
    self.act_bowl = act_bowl
    self.confidence_percent = confidence_percent
  
  def __str__(self):
    return "act_bat:"+str(self.act_bat)+" act_bowl:"+str(self.act_bowl)+" confidence_percent:"+str(self.confidence_percent)

  def __eq__(self, other) : 
    return self.act_bat == other.act_bat and self.act_bowl == other.act_bowl

class GameConfig:
  def __init__(self,horizon_balls,iterative_deepening_depth=1,ucb_mode=None,num_rand_rollout_multiplier=None,non_random_order= np.array(("worst","best","med")),is_use_biased_rollout=False):
    self.horizon_balls = horizon_balls
    self.ucb_mode = ucb_mode
    self.num_rand_rollout_multiplier = num_rand_rollout_multiplier
    self.iterative_deepening_depth = iterative_deepening_depth
    self.non_random_order = non_random_order
    self.is_use_biased_rollout = is_use_biased_rollout