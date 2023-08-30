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
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel,\
    sample_generative_model
    
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
  def __init__(self,belief_points,alpha_vectors,expansions_num,iter_horizon,max_length=100):
    self._belief_points = belief_points
    self._alpha_vectors = alpha_vectors
    self._expansions_num = expansions_num
    self._iter_horizon = iter_horizon
    self._max_length = max_length

  cpdef public plan(self, Agent agent):
      cdef Action action
      cdef float time_taken
      cdef int sims_count

      for _ in range(self._expansions_num):
        for _ in range(self._iter_horizon):
          self._value_backup(self._belief_points,self._alpha_vectors)
        new_beliefs = self._expansion_belief(self._belief_points,self._alpha_vectors)
        self._belief_points = self._belief_points + new_beliefs
      # self._agent = agent   # switch focus on planning for the given agent
      # if not hasattr(self._agent, "tree"):
      #     self._agent.add_attr("tree", None)

      # action, time_taken, sims_count = self._search()
      # self._last_num_sims = sims_count
      # self._last_planning_time = time_taken
      # return action


  cpdef _value_backup(self,list belief_points,list alpha_vectors):
    """
    """
    pass


  cpdef _expansion_belief(self,list belief_points,list alpha_vectors):
    """
    """
    pass
