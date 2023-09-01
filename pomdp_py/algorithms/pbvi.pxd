from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent
from pomdp_py.representations.distribution.histogram cimport Histogram

cdef class PBVI(Planner):
  cdef int _expansions_num
  cdef int _iter_horizon
  cdef list _belief_points
  cdef list _alpha_vectors
  cdef float _discount_factor
  cdef int _max_length
  cdef Agent _agent

  cpdef _argmax(self, list alpha_vectors,Histogram b)

  cpdef _value_backup(self, list belief_points,list prev_alpha_vectors)

  cpdef _expansion_belief(self, list belief_points,list alpha_vectors)

  cpdef _distance(self, list belief_points, Histogram b)