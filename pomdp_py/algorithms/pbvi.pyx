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
import time
import pomdp_py

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

    self._related_actions_index = None
    self._agent = None

  cpdef public plan(self, Agent agent):
      cdef Action action
      cdef float time_taken

      self._agent = agent   # switch focus on planning for the given agent

      start_time = time.time()
      for _ in range(self._expansions_num):
        for _ in range(self._iter_horizon):
          self._alpha_vectors, self._related_actions_index  = self._value_backup(self._belief_points,self._alpha_vectors)

        new_beliefs = self._expansion_belief(self._belief_points,self._alpha_vectors)
        #notice: no prune away the same two belief points.and test that add or not to 
        # other zero alpha vector corrspond to expanded beief point
        self._belief_points = self._belief_points + new_beliefs
        # assert len(self._belief_points) == len(self._alpha_vectors),"the number of the belief\
        # points must be the same with the number of alpha vectors"
        
      time_taken = time.time() - start_time
      print("PBVI's time taken: {}".format(time_taken))

      action = self._best_action(self._alpha_vectors, self._related_actions_index, self._agent.cur_belief)
      return action

  cpdef _best_action(self, list alpha_vectors, list related_actions_index, Histogram cur_b):
    """
    """
    cdef float max_value = float("-inf")
    cdef int index = 0
    cdef int best_alpha_vector_index = 0
    cdef list states = self._agent.transition_model.get_all_states
    cdef float value
    for alpha_vector in alpha_vectors:
      value = 0.0
      for i in range(len(alpha_vector)):
        value += cur_b[states[i]] * alpha_vector[i]
      if value > max_value:
        max_value = value
        best_alpha_vector_index = index

      index += 1

    action = self._agent.policy_model.get_all_actions[related_actions_index[best_alpha_vector_index]]

    return action

  cpdef _argmax(self,list alpha_vectors, Histogram b):
    """
    """
    cdef float max_value = float("-inf")
    cdef list best_alpta_vector
    cdef list states = self._agent.transition_model.get_all_states
    cdef int best_action_index = 0
    cdef int index = 0
    for alpha_vector in alpha_vectors:
      cdef float value = 0.0
      for i in range(len(alpha_vector)):
        value += b[states[i]] * alpha_vector[i]
      if value > max_value:
        max_value = value
        best_alpta_vector = alpha_vector
        best_action_index = index

      index += 1

    return best_alpta_vector,best_action_index

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
          best_vector,_ = self._argmax(acion_observation_alpha_vectors_set[(a,o)],b)  

          assert len(action_set[a]) == best_vector,"Adding vectors must have equal dimensions"

          alpha_vector = list(map(lambda x :x[0]+x[1] ,zip(alpha_vector,best_vector)))
        belief_actions[(index,a)]= alpha_vector
      index += 1
      
    #third step
    cdef list cur_alpha_vectors
    cdef list best_alpha_vector
    cdef int best_action_index = 0
    cdef list actions_related_for_alpha_vectors
    index = 0
    for b in belief_points:
      cdef list vectors
      for a in self._agent.policy_model.get_all_actions:
        vectors.append(belief_actions[(index,a)])
      best_alpha_vector,best_action_index = self._argmax(vectors,b)
      cur_alpha_vectors.append(best_alpha_vector)
      actions_related_for_alpha_vectors.append(best_action_index)
      index += 1

    return cur_alpha_vectors,actions_related_for_alpha_vectors

  cpdef _expansion_belief(self,list belief_points,list alpha_vectors):
    """
    """
    cdef list expanded_belief_points
    for b in belief_points:
      state = b.random()
      cdef float max_dis = float("-inf")
      cdef Histogram expanded_belief
      for action in self._agent.policy_model.get_all_actions:
        next_state, observation, reward, _ = sample_generative_model(self._agent, state, action)
        next_belief = new_belief = pomdp_py.update_histogram_belief(b,
                action, observation,
                self._agent.observation_model,
                self._agent.transition_model)

        if next_belief in belief_points:
          continue
        
        dis = self._distance(belief_points, next_belief)
        if dis > max_dis:
          expanded_belief = next_belief

      expanded_belief_points.append(expanded_belief)

    return expanded_belief_points

  cpdef _distance(self, list belief_points, Histogram b):
    """
    L1 distance
    """
    cdef float min_dis = float("inf")
    for tmp_b in belief_points:
      cdef float dis = 0.0
      for s in self._agent.transition_model.get_all_states:
        dis += abs(tmp_b[s] - b[s])
      if dis < min_dis:
        min_dis = dis

    return min_dis