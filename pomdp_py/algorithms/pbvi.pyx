"""
We implement PBVI(Point-based value iteration) as described in the original paper
Joelle et al. Point-based value iteration: An anytime algorithm for POMDPs. Carnegie Mellon University Robotics Institute. 2003
Joelle et al. Point-based approximations for fast POMDP solving. 2004
"""


"""
PBVI performs best when its belief set is uniformly dense
in the set of reachable beliefs
"""
from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel,RewardModel, GenerativeDistribution, PolicyModel,\
    sample_generative_model
from pomdp_py.representations.distribution.histogram cimport Histogram   

cdef class PBVI(Planner):
  """
  This is a offline  approximations method for POMDP.
  Args:
  belief_points:
  alpha_vectors:
  expansions_num:
  iter_horizon:
  max_length:belif point set size
  """
  def __init__(self,belief_points,alpha_vectors,expansions_num,iter_horizon,discount_factor=0.9,max_length=100):
    self._belief_points = belief_points
    self._alpha_vectors = alpha_vectors
    self._expansions_num = expansions_num
    self._iter_horizon = iter_horizon
    self._discount_factor = discount_factor
    self._max_length = max_length

    self._agent = None

  cpdef public plan(self, Agent agent):
      cdef Action action
      cdef float time_taken
      cdef int sims_count

      self._agent = agent   # switch focus on planning for the given agent

      for _ in range(self._expansions_num):
        for _ in range(self._iter_horizon):
          self._alpha_vectors = self._value_backup(self._belief_points,self._alpha_vectors)

        new_beliefs = self._expansion_belief(self._belief_points,self._alpha_vectors)
        self._belief_points = self._belief_points + new_beliefs

      # action, time_taken, sims_count = self._search()
      # self._last_num_sims = sims_count
      # self._last_planning_time = time_taken
      # return action

  cpdef _argmax(self,list alpha_vectors,Histogram b):
    """
    """
    cdef float max_value = float("-inf")
    cdef list best_alpta_vector
    cdef list states = self._agent.transition_model.get_all_states
    
    for alpha_vector in alpha_vectors:
      cdef float value = 0.0
      for i in range(len(alpha_vector)):
        value += b[states[i]] * alpha_vector[i]
      if value > max_value:
        max_value = value
        best_alpta_vector = alpha_vector

    return best_alpta_vector

  cpdef _value_backup(self,list belief_points,list prev_alpha_vectors):
    """
    """
    #fisrt step
    cdef dict action_set
    cdef float reward
    for a in self._agent.policy_model.get_all_actions:
      cdef list state_rewards
      for s in self._agent.transition_model.get_all_states:
          reward = self._agent.reward_model.sample(s,a)
          state_rewards.append(reward)
      action_set[a] = state_rewards
    
    #time complexity |A||O||V'|*|S|^2
    cdef dict acion_observation_alpha_vectors_set
    for a in self._agent.policy_model.get_all_actions:
      for o in self._agent.observation_model.get_all_observations:         
        cdef list alpha_vectors
        for prev_alpha_vector in prev_alpha_vectors:
          cdef list alpha_vector
          for s in self._agent.transition_model.get_all_states:
            cdef float total_reward  = 0.0
            cdef int state_index = 0
            for s1 in self._agent.transition_model.get_all_states:
              total_reward += self._agent.transition_model.probability(s1,s,a)*self._agent.observation_model.probability(o,s1,a)*prev_alpha_vector[state_index]
              state_index += 1
            alpha_vector.append(self._discount_factor * total_reward)
          alpha_vectors.append(alpha_vector)
        acion_observation_alpha_vectors_set[(a,o)] = alpha_vectors
    
    #second step
    #because b is belief state, b is not suitable as an index
    #so,use integer index
    cdef dict  belief_actions
    cdef int index = 0 
    cdef list alpha_vector
    for b in belief_points:
      for a in self._agent.policy_model.get_all_actions:
        alpha_vector = action_set[a]
        for o in self._agent.observation_model.get_all_observations:  
          cdef list best_vector
          best_vector = self._argmax(acion_observation_alpha_vectors_set[(a,o)],b)  
          assert len(action_set[a]) == best_vector,"Adding vectors must have equal dimensions"
          alpha_vector = list(map(lambda x :x[0]+x[1] ,zip(alpha_vector,best_vector)))
        belief_actions[(index,a)]= alpha_vector
      index += 1
      
    #third step



  cpdef _expansion_belief(self,list belief_points,list alpha_vectors):
    """
    """
    pass
