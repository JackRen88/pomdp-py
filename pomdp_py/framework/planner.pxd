from pomdp_py.framework.basics cimport Action, Agent, Observation, State

cdef class Planner:
    cdef str _name
    cpdef public plan(self, Agent agent)
    cpdef public update(self, Agent agent, Action real_action, Observation real_observation)
    
