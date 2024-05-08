"""The classic crying baby problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

The description of the crying baby problem is as follows: (Quote from
`Partially Observable MDPs
<https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/notebooks/2-POMDPs.jl>`_ by
Julia Academy course )



States: hungry, full
Actions: feed, ignore
Observations: crying,quiet
Rewards:
    -10 for hungry. 
    -5 for feed (cost isn't too small(-5),because feed when crying appear suddenly after ignoring less times for full ,that is bad for baby,too full)-->not be decided
    0.0 for other
Transition:
    becomes hungry:0.1
Observation:
    crying when hungry: 0.8
    crying when full: 0.1
    
Note that in this example, the CryingBabyProblem is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more
complicated examples.)
"""

import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys


class CryingBabyState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
        assert self.name in {
            "hungry", "full"}, "state is invalid,%s" % self.name

    # dic request
    def __hash__(self):
        return hash(self.name)
    # dic request

    def __eq__(self, other):
        if isinstance(other, CryingBabyState):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "CryingBabyState(%s)" % self.name
    # def other(self):
    #     if self.name.endswith("hungry"):
    #         return CryingBabyState("full")
    #     else:
    #         return CryingBabyState("hungry")


class CryingBabyAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
        assert self.name in {
            "feed", "ignore"}, "action is invalid,%s" % self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, CryingBabyAction):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "CryingBabyAction(%s)" % self.name


class CryingBabyObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
        assert self.name in {
            "crying", "quiet"}, "Observation is invalid,%s" % self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, CryingBabyObservation):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "CryingBabyObservation(%s)" % self.name

# Observation model


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise_when_hungry=0.2, noise_when_full=0.1):
        self.noise_when_hungry = noise_when_hungry
        self.noise_when_full = noise_when_full

    def probability(self, observation, next_state, action):
        if next_state.name == "hungry":
            if observation.name == "crying":
                return 1.0 - self.noise_when_hungry
            else:
                return self.noise_when_hungry
        else:
            if observation.name == "crying":
                return self.noise_when_full
            else:
                return 1 - self.noise_when_full

    def sample(self, next_state, action):
        if next_state.name == "hungry":
            thresh = self.noise_when_hungry
            if random.uniform(0, 1) < thresh:
                return CryingBabyObservation("quiet")
            else:
                return CryingBabyObservation("crying")
        else:
            thresh = self.noise_when_full
            if random.uniform(0, 1) < thresh:
                return CryingBabyObservation("crying")
            else:
                return CryingBabyObservation("quiet")

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [CryingBabyObservation(s)
                for s in {"crying", "quiet"}]

# Transition Model


class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if state.name.startswith("hungry") and action.name.startswith("feed"):
            if next_state.name.startswith("hungry"):
                return 0.0
            else:
                return 1.0
        if state.name.startswith("hungry") and action.name.startswith("ignore"):
            if next_state.name.startswith("hungry"):
                return 1.0
            else:
                return 0.0
        if state.name.startswith("full") and action.name.startswith("feed"):
            if next_state.name.startswith("hungry"):
                return 0.0
            else:
                return 1.0
        if state.name.startswith("full") and action.name.startswith("ignore"):
            if next_state.name.startswith("hungry"):
                return 0.1
            else:
                return 0.9

    def sample(self, state, action):
        if action.name.startswith("feed"):
            return CryingBabyState("full")
        else:
            if state.name.startswith("hungry"):
                return CryingBabyState("hungry")
            else:
                thresh = 0.1
                if random.uniform(0, 1) < thresh:
                    return CryingBabyState("hungry")
                else:
                    return CryingBabyState("full")

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [CryingBabyState(s) for s in {"hungry", "full"}]

# Reward Model


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        return (-10 if state.name.startswith("hungry") else 0) + (-8 if action.name.startswith("feed") else 0)

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

# Policy Model


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    ACTIONS = [CryingBabyAction(s)
               for s in {"feed", "ignore"}]

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class CryingBabyProblem(pomdp_py.POMDP):
    """
    In fact, creating the class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, noise_when_hungry, noise_when_full, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(
                                   noise_when_hungry, noise_when_full),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="CryingBabyProblem")

    @staticmethod
    def create(state="full", belief=0.5, noise_when_hungry=0.2, noise_when_full=0.1):
        """
        Args:
            state (str): could be 'hungry' or 'full';
                         True state of the environment
            belief (float): Initial belief. Between 0-1.
            noise_when_hungry(float): Noise for the observation when hungry
            noise_when_full (float): Noise for the observation when full

        """
        init_true_state = CryingBabyState(state)
        init_belief = pomdp_py.Histogram({
            CryingBabyState("hungry"): belief,
            CryingBabyState("full"): 1.0 - belief
        })
        crying_baby_problem = CryingBabyProblem(noise_when_hungry, noise_when_full,
                                                init_true_state, init_belief)
        crying_baby_problem.agent.set_belief(init_belief, prior=True)
        return crying_baby_problem


def test_planner(crying_baby_problem, planner, nsteps=3,
                 debug_tree=False):
    """
    Runs the action-feedback loop of CryingBaby problem POMDP

    Args:
        crying_baby_problem (CryingBabyProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(crying_baby_problem.agent)
        if debug_tree:
            # from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(crying_baby_problem.agent.tree)
            import pdb
            pdb.set_trace()

        if crying_baby_problem.env.state.name.startswith("full") and action.name.startswith("feed"):
            """
            make it clearer that baby is too full.
            """
            print("\n")

        print("==== Step %d ====" % (i+1))
        print("True current env state:", crying_baby_problem.env.state)
        print("Belief:", crying_baby_problem.agent.cur_belief)
        print("Action:", action)
        # There is state transition for the crying baby domain.
        # In general, the environment state can be transitioned
        # using
        #
        #   reward = crying_baby_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        reward = crying_baby_problem.env.state_transition(action, execute=True)
        print("Reward:", reward)
        print("True next env state:", crying_baby_problem.env.state)

        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this real observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = crying_baby_problem.agent.observation_model.sample(crying_baby_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that here crying_baby_problem.env.state stores the
        # environment state after action execution
        real_observation = crying_baby_problem.agent.observation_model.sample(
            crying_baby_problem.env.state, action)
        print(">> Observation:",  real_observation)
        crying_baby_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        # TODO(all):here select method of update belief according to planner
        planner.update(crying_baby_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(crying_baby_problem.agent.cur_belief,
                      pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                crying_baby_problem.agent.cur_belief,
                action, real_observation,
                crying_baby_problem.agent.observation_model,
                crying_baby_problem.agent.transition_model)
            crying_baby_problem.agent.set_belief(new_belief)
            # print("new_belief: ",new_belief)


def main():
    init_true_state = random.choice([CryingBabyState("hungry"),
                                     CryingBabyState("full")])
    init_belief = pomdp_py.Histogram({CryingBabyState("hungry"): 0.5,
                                      CryingBabyState("full"): 0.5})
    crying_baby_problem = CryingBabyProblem(0.2, 0.1,  # when full or hungry observation noise
                                            init_true_state, init_belief)

    # print("** Testing value iteration **")
    """     
        horizon is too large to compute expensive ,
        because layer level of tree is too big,
        resulting in getting too much policy tree,so need to prune tree.
        here in general,horizon = 3
    """
    # vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    # test_planner(crying_baby_problem, vi, nsteps=10)

    print("** Testing point-based value iteration **")
    crying_baby_problem.agent.set_belief(init_belief, prior=True)

    pbvi = pomdp_py.PBVI(belief_points=[init_belief], alpha_vectors=[[0 for _ in range(len(init_belief))]],
                         expansions_num=20, iter_horizon=50, discount_factor=0.95, max_length=100)

    test_planner(crying_baby_problem, pbvi, nsteps=10)

    # print("\n** Testing POUCT **")
    # Reset agent belief
    # crying_baby_problem.agent.set_belief(init_belief, prior=True)
    # pouct = pomdp_py.POUCT(max_depth=5, discount_factor=0.95,
    #                        num_sims=4096, exploration_const=50,
    #                        rollout_policy=crying_baby_problem.agent.policy_model,
    #                        show_progress=True)
    # test_planner(crying_baby_problem, pouct, nsteps=10)
    # TreeDebugger(tiger_problem.agent.tree).pp

    # # Reset agent belief
    # crying_baby_problem.agent.set_belief(init_belief, prior=True)
    # crying_baby_problem.agent.tree = None

    # print("** Testing POMCP **")
    # crying_baby_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True)
    # pomcp = pomdp_py.POMCP(max_depth=5, discount_factor=0.95,
    #                        num_sims=3000, exploration_const=50,
    #                        rollout_policy=crying_baby_problem.agent.policy_model,
    #                        show_progress=True, pbar_update_interval=500)
    # test_planner(crying_baby_problem, pomcp, nsteps=10)
    # TreeDebugger(tiger_problem.agent.tree).pp


if __name__ == '__main__':
    main()
#print hello world
