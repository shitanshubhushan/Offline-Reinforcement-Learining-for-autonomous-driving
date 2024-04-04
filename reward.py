import numpy as np

def reward(r_p, r_v, r_C, p, p_hat, v, v_hat, C, next_steps):
  '''Reward function for RL system
  INPUTS: 
          r_p: Hyperparameter, factor for position term in reward equation
          r_v: Hyperparameter, factor for velocity term in reward equation
          r_C: Hyperparameter, factor for collision term in reward equation
          p: Actual position of objects over next_steps
          p_hat: RL agent planned position of ego over next_steps
          v: Actual velocity of ego over next_steps
          v_hat: RL agent planned velocity of ego over next_steps
          C: Number of collisions over next_steps
          next_steps: Hyperparameter: Number of timesteps to consider in application of reward function
  OUTPUTS: 
          R: total reward'''
  
  # How to bound the R_p and R_v terms is something we should investigate in our training iterations

  p = p[0:next_steps-1]
  p_hat = p_hat[0:next_steps-1]
  v = v[0:next_steps-1]
  v_hat = v_hat[0:next_steps-1]

  R_p = r_p/(np.norm(p-p_hat)**2) #This needs to be bounded per the feedback on our progress report
  R_p = np.max(R_p, 100) #Bound to 100 (should we make this a hyperparameter?)

  R_v = r_v/(np.norm(v-v_hat)**2) #This needs to be bounded per the feedback on our progress report
  R_v = np.max(R_v, 100) #Bound to 100 (should we make this a hyperparameter?)

  R_C = -r_C*np.sum(C)

  R = R_p + R_v + R_C
  
  return R