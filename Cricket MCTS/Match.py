import numpy as np
import time
from env import *

class Match:

  def __init__(self,num_balls, explore_num_balls,action_timeout, explore_timeout, TeamOne, TeamTwo):
    self.environment = Environment()
    self.num_balls = num_balls
    self.explore_num_balls = explore_num_balls
    self.action_timeout = action_timeout          # time limit for any given act
    self.explore_timeout = explore_timeout        # time limit for exploration phase
    #feature[0] : batting average, feature[1]: strike-rate, feature[2]: bowling average, feature[3]: economy
    
    # Weak opponent
    # self.team_one_features = np.ones((5,4),dtype=int) 
    # self.team_two_features = np.ones((5,4),dtype=int)*5
    # Strong opponent
    # self.team_one_features = np.ones((5,4),dtype=int)*5
    # self.team_two_features = np.ones((5,4),dtype=int)
    #Medium case
    self.team_one_features = np.ones((5,4),dtype=int)*3 
    self.team_two_features = np.ones((5,4),dtype=int)*3

    self.team_one = TeamOne(self.team_one_features,self.team_two_features,self.environment)
    self.team_two = TeamTwo(self.team_two_features,self.team_one_features,self.environment)
    self.current_batters_list = np.array([1,1,1,1,1])  # a coordinate is set to 0 when that corresponding batter gets out, if the third player is out then you have [1,1,0,1,1]
    self.current_bowlers_list = np.array([2,2,2,2,2])  # if the 4th bowler bowls the first over, after first over we have [2,2,2,1,2].
    self.num_miss_team_batting = 0
    self.num_miss_team_bowling = 0


  def explore_phase_team(self,team_id):    #timed phase
    if (team_id == 1):
      team = self.team_one
    else:
      team = self.team_two
    start_time      = time.time()
    team.explore(self.explore_num_balls)    
    end_time = time.time()    
    if(end_time - start_time > self.explore_timeout):  
      print("Timing Violation During Exploration Phase")
      

  def explore_phase(self):     #allow each of the teams to explore
    print("Team 1 explore")
    self.explore_phase_team(1)
    print("Team 2 explore")
    self.explore_phase_team(2)


  def get_valid_bowler(self,next_bowler):
    if (self.current_bowlers_list[next_bowler]==0):
      print("Bowler Invalid, Choosing Random Bowler")
      bowlers_with_overs_left = np.where(self.current_bowlers_list>0)[0]
      # next_bowler = np.random.choice(bowlers_with_overs_left) # For phase 2
      next_bowler = bowlers_with_overs_left[0]  # Sequentially send bowlers
    return next_bowler


  def get_valid_batter(self,next_batter):
    if (self.current_batters_list[next_batter]==0):
      print("Batter Invalid, Choosing Random Batter"+str(next_batter)+" current_batters_list:"+str(self.current_batters_list))
      batters_not_out = np.where(self.current_batters_list>0)[0]
      next_batter = np.random.choice(batters_not_out)
    return next_batter

  def next_batter(self,team_batting):
    next_batter = team_batting.get_next_batter()
    next_batter = self.get_valid_batter(next_batter)
    feature_batter = team_batting.self_features[next_batter,0:2]
    return next_batter,feature_batter


  def next_bowler(self,team_bowling):
    # next_bowler = team_bowling.get_next_bowler() # For phase 2
    next_bowler = self.get_valid_bowler(0)
    feature_bowler = team_bowling.self_features[next_bowler,2:4]
    return next_bowler, feature_bowler

  def get_team_batting_action(self,team_batting,ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,current_batter_index,current_bowler_index,random=False,iterative_deepening_depth=6,ucb_mode=None,num_rand_rollout_multiplier=None):
    start_time      = time. time()
    if(random):
      batting_action  = team_batting.get_random_batting_action(ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,self.action_timeout)
    else:
      batting_action  = team_batting.get_batting_action(ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,current_batter_index,current_bowler_index,self.action_timeout,iterative_deepening_depth=iterative_deepening_depth,ucb_mode=ucb_mode,num_rand_rollout_multiplier=num_rand_rollout_multiplier)
    end_time        = time. time()
    if(end_time - start_time > self.action_timeout):
      batting_action = 0 #this is the default option, we have to fix the penalisation strategy
      self.num_miss_team_batting = self.num_miss_team_batting + 1
      print("Action timeout crossed===================================================================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      # raise Exception("Action timeout crossed===================================================================")
    return batting_action


  def get_team_bowling_action(self,team_bowling,ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,current_batter_index,current_bowler_index):
    start_time      = time. time()
    bowling_action  = team_bowling.get_bowling_action(ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,current_batter_index,current_bowler_index)
    end_time        = time. time()
    if(end_time - start_time > self.action_timeout):
      bowling_action = 0 #this is the default option, we have to fix the penalisation strategy
      self.num_miss_team_bowling = self.num_miss_team_bowling + 1
    return bowling_action



  def innings(self,innigins_id,runs_to_chase,random=False,iterative_deepening_depth=6,ucb_mode='LOOSE',num_rand_rollout_multiplier=None):
    total_runs = 0
    wickets_left = 5
    self.current_batters_list =np.array([1,1,1,1,1])
    self.current_bowlers_list =np.array([2,2,2,2,2])
    self.num_miss_team_batting = 0
    self.num_miss_team_bowling = 0
    if (innigins_id == 1 ):
      team_batting = self.team_one
      team_bowling = self.team_two
    else:
      team_batting = self.team_two
      team_bowling = self.team_one
    # Initialising the first batter and first bowler
    batter, feature_batter = self.next_batter(team_batting)
    bowler, feature_bowler = self.next_bowler(team_bowling)
    
    for ball in range(self.num_balls):
      if np.sum(self.current_batters_list) > 0 :
        batting_action = self.get_team_batting_action(team_batting,ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,batter,bowler,random,iterative_deepening_depth=iterative_deepening_depth,ucb_mode=ucb_mode,num_rand_rollout_multiplier=num_rand_rollout_multiplier)
        bowling_action = self.get_team_bowling_action(team_bowling,ball,total_runs,wickets_left,runs_to_chase,feature_batter,feature_bowler,batter,bowler)
        wicket, runs   = self.environment.get_outcome(feature_batter, feature_bowler, batting_action, bowling_action)
        total_runs     = total_runs + runs
        if (wicket > 0):
          self.current_batters_list[batter] = 0
          wickets_left = wickets_left - 1
          if(np.sum(self.current_batters_list) > 0 ):
            batter,feature_batter = self.next_batter(team_batting)
        if ((ball + 1)%6 ==0 ):
          self.current_bowlers_list[bowler] = self.current_bowlers_list[bowler]-1
          if(np.sum(self.current_bowlers_list) > 0 ) :
            bowler, feature_bowler = self.next_bowler(team_bowling)
        print("Actual score in this ball. Wicket: "+str(wicket)+" run: "+str(runs)+" total runs: "+str(total_runs)+" wickets left: "+str(wickets_left))

    return total_runs, self.current_batters_list, self.current_bowlers_list, self.num_miss_team_batting, self.num_miss_team_bowling
