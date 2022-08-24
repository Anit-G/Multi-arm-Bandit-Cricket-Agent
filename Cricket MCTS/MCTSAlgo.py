import numpy as np
import time
import statistics
from MCTSNode import *
from GameLogic import *

class MCTSAlgo:
  def __init__(self,environment,team,game_config):
    self.environment = environment
    self.my_team = team
    self.game_config = game_config
    
  def get_next_non_random_order(self,current_non_random):
    next_index = np.where(self.game_config.non_random_order == current_non_random )[0][0] + 1

    if(next_index == len(self.game_config.non_random_order)):
      return self.game_config.non_random_order[0]
    return self.game_config.non_random_order[next_index]

  def run(self,state,overall_action_timeout,my_role,biased_rollout_dict):
    overall_start_time =  time. time()
    gameLogic = GameLogic(self.environment,self.my_team,self.game_config,biased_rollout_dict)
    root = MCTSNode(None,'me_to_play',state)
    root.expand_children(gameLogic)
    current_time =  time. time()
    iterative_deepening_size= self.game_config.iterative_deepening_depth
    actual_horizon = self.game_config.horizon_balls
    horizon_per_ID = []
    iterative_deepening_round_count = 0
    current_horizon = 0
    while(current_horizon<actual_horizon):
      iterative_deepening_round_count += 1
      current_horizon = root.state.num_ball + iterative_deepening_round_count * iterative_deepening_size
      if(current_horizon > actual_horizon):
        horizon_per_ID.append(actual_horizon)
      else:
        horizon_per_ID.append(current_horizon)
    
    action_timeout = overall_action_timeout/len(horizon_per_ID)
    iterative_depth_count = 0
    time_left = (overall_action_timeout - (current_time -overall_start_time))
    time_taken_for_loop = 0.0000001
    # If time left is less than twice the time for previous simulation, then exit
    while(True):
      if(iterative_depth_count == len(horizon_per_ID)):
        break
      elif(iterative_depth_count == (len(horizon_per_ID)- 1)):
        current_time =  time. time()
        action_timeout = (overall_action_timeout - (current_time - overall_start_time))
      simulation_count = 0
      current_sim_depth = 0
      all_simulation_depth = []

      if(not(within_time_constraints(overall_start_time,time_taken_for_loop,overall_action_timeout))):
        break
      
      current_horizon = horizon_per_ID[iterative_depth_count]
      self.game_config.horizon_balls = current_horizon
      
      start_time =  time. time()
      iterative_depth_count+=1
      print("In iterative deepening round number ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"+str(iterative_depth_count)+"   time_taken_for_loop: "+str(time_taken_for_loop)+" Horizon ID: "+str(horizon_per_ID)+" Actiontimeout: "+str(action_timeout))
      while(True):
      # while(simulation_count < 100):
        if(not(within_time_constraints(start_time,time_taken_for_loop,action_timeout))):
          break

        simulation_count += 1
        # print("*********************************************")
        # print("Simulation count:",simulation_count)
        start_of_loop_time = time.time()

        next_best_child = root
        next_best_child.visit_count += 1
        current_sim_depth = 1
        # Selection phase
        while(next_best_child.is_node_expanded() and not(gameLogic.is_state_terminal(next_best_child.state))):
          current_sim_depth += 1
          if(next_best_child.parent is None):
            next_best_child = next_best_child.select_best_child(gameLogic,round_num=simulation_count)
          else:
            next_best_child = next_best_child.select_best_child(gameLogic)
          # print("!!!!!!!!!!!!!!!!!!!!!!!!! Selection phase !!!!!!!!!!!!!!!!!!!!!"+str(next_best_child))
        
        # print("Selection phase complete. Best child",next_best_child)
        if(not(within_time_constraints(start_time,time_taken_for_loop,action_timeout))):
          simulation_count -= 1
          break

        #Expansion phase
        next_best_child.expand_children(gameLogic)
        
        # Simulation phase
        # Random rollout done only once is bad since among huge state space only one path is sampled for value at horizon.
        # So one heuristic is to increase the number of rollouts when depth of selection increases.(intuition is when depth increases MCTS is confident about that path. Hence explore that more and get actual value)
        number_of_random_rollouts = 2
        if(self.game_config.num_rand_rollout_multiplier is None):
          number_of_random_rollouts = 2
        else:
          number_of_random_rollouts = 1 + current_sim_depth * self.game_config.num_rand_rollout_multiplier
        
        sum_runs_horizon = 0
        if(number_of_random_rollouts > 6):
          number_of_non_random_rr = len(self.game_config.non_random_order)
          current_non_random_mode = self.game_config.non_random_order[0]
          wicket_left_at_horizon, runs_at_horizon = next_best_child.simulate_random_rollout(gameLogic,current_non_random_mode)
          sum_runs_horizon += runs_at_horizon
          for i in range(1,number_of_non_random_rr):
            current_non_random_mode = self.get_next_non_random_order(current_non_random_mode)
            wicket_left_at_horizon, runs_at_horizon = next_best_child.simulate_random_rollout(gameLogic,current_non_random_mode)
            sum_runs_horizon += runs_at_horizon
          
          for i in range((number_of_random_rollouts - number_of_non_random_rr)):
            wicket_left_at_horizon, runs_at_horizon = next_best_child.simulate_random_rollout(gameLogic)
            sum_runs_horizon += runs_at_horizon
        else:
          for i in range(number_of_random_rollouts):
            wicket_left_at_horizon, runs_at_horizon = next_best_child.simulate_random_rollout(gameLogic)
            sum_runs_horizon += runs_at_horizon
        
        avg_runs_horizon = sum_runs_horizon / number_of_random_rollouts
        # Calculate the run per ball and use that as value of node
        # print("avg_runs_horizon: "+str(avg_runs_horizon)+" Is terminal?: "+str(gameLogic.is_state_terminal(next_best_child.state)))

        avg_runs_per_ball = avg_runs_horizon/(self.game_config.horizon_balls - root.state.num_ball)
        
        next_best_child.update_value_of_node(avg_runs_per_ball)
        # print("Simulation phase complete. Runs at horizon:",runs_at_horizon)

        if(not(within_time_constraints(start_time,time_taken_for_loop,action_timeout))):
          simulation_count -= 1
          break

        # Backpropogate node value phase
        value_list = next_best_child.back_propogate_node_value(gameLogic)
        # print("Back propogation phase complete. Value list is: ",value_list)

        current_time =  time. time()
        time_taken_for_loop = current_time - start_of_loop_time

        #Monitor depths reached by simulations for analysis
        all_simulation_depth.append(current_sim_depth)
      
      max_sim_depth = max(all_simulation_depth)
      pos_max_sim__depth = [i for i, j in enumerate(all_simulation_depth) if j == max_sim_depth]
      depth_median = statistics.median(all_simulation_depth)
      depth_mean = statistics.mean(all_simulation_depth)
      current_time =  time. time()
      time_left = (action_timeout - (current_time - start_time))
      # print("root node: "+str(root))
      # print("Children of root node: ")
      # for each_c  in root.children:
      #   print("^^^^^^^^^^^^^^^^^^^")
      #   print(each_c)
      #   # for eac  in each_c.children:
      #   #   print("####################")
      #   #   print(eac)
      #   #   print("####################")
      #   print("^^^^^^^^^^^^^^^^^^^")
        # print("Act_bat:"+str(each_c.state.batting_action)+" ValueOfNode:"+str(each_c.valueOfNode)+" Visit count:"+str(each_c.visit_count)+" UCB score:"+str(each_c.ucb_scoreOfNode)+" Runs:"+str(each_c.state.total_runs))
      # print("Current horizon:"+str(current_horizon)+" Current action timeout:"+str(action_timeout))
      print("Total number of simulations: "+str(simulation_count)+" time taken in last loop: "+str(time_taken_for_loop)+" Time left:"+str(time_left) +" Time utilized:"+str(current_time -start_time)+" \n Max depth traversed by simulation: "+str(max_sim_depth)+" simulation position of max depth: "+str(pos_max_sim__depth)+"\n Simulation depth median: "+str(depth_median)+" Simulation depth mean:"+str(depth_mean))
    
    current_time =  time. time()
    time_left = (overall_action_timeout - (current_time - overall_start_time))
    # print("Root node: "+str(root))
    print("Overall Time left:"+str(time_left) +" Overall Time utilized:"+str(current_time - overall_start_time))

    if(gameLogic.game_config.is_use_biased_rollout):
      return self.get_action(gameLogic,root,my_role,simulation_count),gameLogic.biased_rollout_dict
    else:
      return self.get_action(gameLogic,root,my_role,simulation_count),None

  def get_action(self,gameLogic,root_node,my_role,simulation_count):
    best_child = root_node.select_best_child(gameLogic,round_num = simulation_count)
    if(my_role == 'batting'):
      return best_child.state.batting_action
    elif(my_role == 'bowling'):
      return best_child.state.bowling_action
