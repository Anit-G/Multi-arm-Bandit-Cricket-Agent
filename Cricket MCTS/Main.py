from Match import *
from Team import *

if __name__ == '__main__':
    # setting time outs
    explore_timeout = 10
    action_timeout = 10

    explore_num_balls = 60 # this number will be much lesser when we actually test the code.
    num_balls = 60         # fixed
    match = Match(num_balls, explore_num_balls,action_timeout, explore_timeout,Australia,India)
    match.explore_phase()
    first_innings_score_rand, batters_list, bowlers_list, num_miss_team_batting, num_miss_team_bowling = match.innings(1,float('inf'),random=True)
    print("randomly played Score of innings: "+str(first_innings_score_rand))
    first_innings_score, batters_list, bowlers_list, num_miss_team_batting, num_miss_team_bowling = match.innings(1,float('inf'),iterative_deepening_depth=4,ucb_mode='LOOSE',)
    print("Score of innings: "+str(first_innings_score))
    print("randomly played Score of innings: "+str(first_innings_score_rand))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ SECOND INNINGS START @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # second_innings_score, batters_list, bowlers_list, num_miss_team_batting, num_miss_team_bowling = match.innings(2,first_innings_score)