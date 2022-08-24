import numpy as np
class Environment: 
  # def get_outcome_random(self,feature_batter, feature_bowler, batting_action, bowling_action):
  #   wicket = np.random.randint(0,2)
  #   runs = np.random.randint(0,7)
  #   return wicket, runs

  def get_outcome(self,feature_batter, feature_bowler, batting_action, bowling_action,mode='default'):
    """
    Calculate Pout and Pruns. 
    The proportional realations of pruns and pout are calculated using the state values.
    These are then used as a mean value for a normal distribution from which the 
    probablities are sampled. The Variance for each distribution is the same at 0.135.

    Wickets: the value of pout is rounded to the nearest integer (0 or 1).
    Runs: the value of pruns is scaled using batting action then rounded to nearest integer.

    Initial beta distribution is shifted such that avg runs is 3
        Args:-------------------------------------
        feature_batter: batting features of the batter (shape = (2,1))
        feature_bowler: bowling features of the bowler (shape = (2,1))
        batting_action: int between 0 and 6
        bowling_action: int between 1 and 3
        
        
        Returns:----------------------------------
          wickets: int
          runs: int      
    """
    convert = {'e':1,'n':2,'a':3}
    #Define all state values:
    try:
      actbat = batting_action
      actbowl = bowling_action
      avgbat = feature_batter[0]
      sr = feature_batter[1]
      avgbowl = feature_bowler[0]
      eco = feature_bowler[1]
    except:
      print("Error>>>>>>>>>>>>>>>> FBowler:"+str(feature_bowler)+" FBatter: "+str(feature_batter)+" ABowl: "+str(bowling_action)+" Abat:"+str(batting_action))
    
    if isinstance(actbowl,str):
      actbowl = convert[actbowl]

    # Find pruns
    # Define Beta distribution a and b
    mean_runs = actbowl*eco/sr/15
    #mean runs can't ever be 1 if it does n=a=b=0
    mean_runs = min(0.95,mean_runs)
    s = 0.16666667
    n = mean_runs*(1-mean_runs)/s**2
    
    a = mean_runs*n
    if a<=0:
      print(f'a:{a}')
      print(f'n:{n}')
      print(f'mean_runs:{mean_runs}')
      print(f'actbowl:{actbowl}')
    b = n*(1-mean_runs)
    pruns = np.random.beta(a,b)+0.2
    # if actbat in [4,6]:
    #   pruns+=0.1
    # if actbat==3:
    #   pruns-=0.1
    pruns = max(0, min(1, pruns))
    r = int(np.round(actbat*pruns))

    # Find pouts
    # Define a bernoulli dist
    actbat = np.array([1,2,3,4,5,6,6])[actbat]
    mean_outs = (actbat*actbowl*avgbat/avgbowl)/90
    mean_outs = max(0.1, min(1, mean_outs))
    pout = np.random.binomial(1,mean_outs)
    w = int(np.round(pout))

    # check the range of the value of wicket
    w = max(0,min(1,w))
    if r<0:
      r=0
    if r==5:
      r=4
    
    # Remove this code later. Only for testing performance
    # if(actbat == 6):
    #   return 0,r
      
    return w,r
            
