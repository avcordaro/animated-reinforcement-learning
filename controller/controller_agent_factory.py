from controller.controller_value_iter import ControllerValueIter
from controller.controller_policy_iter import ControllerPolicyIter
from controller.controller_first_visit_mc import ControllerFirstVisitMC
from controller.controller_q_learning import ControllerQLearning
from controller.controller_sarsa import ControllerSARSA
from controller.controller_reinforce import ControllerREINFORCE
from controller.controller_actor_critic import ControllerActorCritic
from controller.controller_dqn_cartpole import ControllerDQNCartPole
from controller.controller_dqn_pong import ControllerDQNPong
from controller.controller_ddpg_lunar_lander import ControllerDDPGLunarLander
from controller.controller_ddpg_bipedal_walker import ControllerDDPGBipedalWalker
from model.agent_value_iter import AgentValueIter
from model.agent_policy_iter import AgentPolicyIter
from model.agent_first_visit_mc import AgentFirstVisitMC
from model.agent_q_learning import AgentQLearning
from model.agent_sarsa import AgentSARSA
from model.agent_reinforce import AgentREINFORCE
from model.agent_actor_critic import AgentActorCritic
from model.agent_dqn_cartpole import AgentDQNCartPole
from model.agent_dqn_pong import AgentDQNPong
from model.agent_ddpg_lunar_lander import AgentDDPGLunarLander
from model.agent_ddpg_bipedal_walker import AgentDDPGBipedalWalker


class ControllerAgentFactory:
    """
    Factory class used for instantiating the correct Controller and Agent implementation, based
    on the algorithm selected on the GUI.
    """

    def create_controller_and_agent(algorithm, env, gui):
        # Ensures the "Load best model" checkbutton is only visible for DQN algorithms.
        gui.load_best_model_chkbtn.grid_remove()
        if algorithm == "Value Iteration":
            agent = AgentValueIter(env)
            return ControllerValueIter(env, agent, gui), agent
        if algorithm == "Policy Iteration":
            agent = AgentPolicyIter(env)
            return ControllerPolicyIter(env, agent, gui), agent
        if algorithm == "First-visit Monte Carlo Control":
            agent = AgentFirstVisitMC(env)
            return ControllerFirstVisitMC(env, agent, gui), agent
        if algorithm == "Q-Learning":
            agent = AgentQLearning(env)
            return ControllerQLearning(env, agent, gui), agent
        if algorithm == "SARSA":
            agent = AgentSARSA(env)
            return ControllerSARSA(env, agent, gui), agent
        if algorithm == "REINFORCE":
            agent = AgentREINFORCE(env)
            return ControllerREINFORCE(env, agent, gui), agent
        if algorithm == "Actor-Critic":
            agent = AgentActorCritic(env)
            return ControllerActorCritic(env, agent, gui), agent
        if algorithm == "Deep Q-Network for CartPole":
            agent = AgentDQNCartPole(env)
            gui.load_best_model_chkbtn.grid()
            gui.load_best_model.set(0)
            return ControllerDQNCartPole(env, agent, gui), agent
        if algorithm == "Deep Q-Network for Pong":
            agent = AgentDQNPong(env)
            gui.load_best_model_chkbtn.grid()
            gui.load_best_model.set(0)
            return ControllerDQNPong(env, agent, gui), agent
        if algorithm == "DDPG for Lunar Lander":
            agent = AgentDDPGLunarLander(env)
            gui.load_best_model_chkbtn.grid()
            gui.load_best_model.set(0)
            return ControllerDDPGLunarLander(env, agent, gui), agent
        if algorithm == "DDPG for Bipedal Walker":
            agent = AgentDDPGBipedalWalker(env)
            gui.load_best_model_chkbtn.grid()
            gui.load_best_model.set(0)
            return ControllerDDPGBipedalWalker(env, agent, gui), agent
