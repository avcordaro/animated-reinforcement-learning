from model.environment_frozenlake import FrozenLake
from model.environment_cliff_walking import CliffWalking
from model.environment_taxi_driver import TaxiDriver
from model.environment_cartpole import CartPole
from model.environment_pong import Pong
from model.environment_lunar_lander import LunarLander
from model.environment_bipedal_walker import BipedalWalker
from view.animation_frozenlake import AnimationFrozenLake
from view.animation_frozenlake8x8 import AnimationFrozenLake8x8
from view.animation_cliff_walking import AnimationCliffWalking
from view.animation_taxi_driver import AnimationTaxiDriver
from view.animation_cartpole import AnimationCartPole
from view.animation_pong import AnimationPong
from view.animation_lunar_lander import AnimationLunarLander
from view.animation_bipedal_walker import AnimationBipedalWalker


class EnvironmentAnimationFactory:
    """
    Factory class used for instantiating the correct Environment and Animation implementation,
    based on the environment selected on the GUI.
    """

    def create_environment_and_animation(environment, root):
        if environment == "Frozen Lake 4x4":
            env = FrozenLake(False, grid_size=4)
            animation = AnimationFrozenLake(root, env.GRID_MAP, update_animation=True)
            algorithms = ["Value Iteration", "Policy Iteration", "First-visit Monte Carlo Control", "Q-Learning",
                          "SARSA", "REINFORCE", "Actor-Critic"]
            return env, animation, algorithms
        if environment == "Frozen Lake 4x4 - Slippery":
            env = FrozenLake(True, grid_size=4)
            animation = AnimationFrozenLake(root, env.GRID_MAP, update_animation=True)
            algorithms = ["Value Iteration", "Policy Iteration", "First-visit Monte Carlo Control", "Q-Learning",
                          "SARSA", "REINFORCE", "Actor-Critic"]
            return env, animation, algorithms
        if environment == "Frozen Lake 8x8":
            env = FrozenLake(False, grid_size=8)
            animation = AnimationFrozenLake8x8(root, env.GRID_MAP, update_animation=True)
            algorithms = ["Value Iteration", "Policy Iteration", "First-visit Monte Carlo Control", "Q-Learning",
                          "SARSA", "REINFORCE", "Actor-Critic"]
            return env, animation, algorithms
        if environment == "Frozen Lake 8x8 - Slippery":
            env = FrozenLake(True, grid_size=8)
            animation = AnimationFrozenLake8x8(root, env.GRID_MAP, update_animation=True)
            algorithms = ["Value Iteration", "Policy Iteration", "First-visit Monte Carlo Control", "Q-Learning",
                          "SARSA", "REINFORCE", "Actor-Critic"]
            return env, animation, algorithms
        if environment == "Cliff Walking":
            env = CliffWalking()
            animation = AnimationCliffWalking(root, env.GRID_MAP, update_animation=True)
            algorithms = ["Q-Learning", "SARSA"]
            return env, animation, algorithms
        if environment == "Taxi Driver":
            env = TaxiDriver()
            animation = AnimationTaxiDriver(root, env, update_animation=True)
            algorithms = ["Value Iteration", "Policy Iteration", "Q-Learning", "SARSA"]
            return env, animation, algorithms
        if environment == "CartPole":
            env = CartPole()
            animation = AnimationCartPole(root, env, update_animation=True)
            algorithms = ["Deep Q-Network for CartPole"]
            return env, animation, algorithms
        if environment == "Pong":
            env = Pong()
            animation = AnimationPong(root, env, update_animation=True)
            algorithms = ["Deep Q-Network for Pong"]
            return env, animation, algorithms
        if environment == "Lunar Lander":
            env = LunarLander()
            animation = AnimationLunarLander(root, env, update_animation=True)
            algorithms = ["DDPG for Lunar Lander"]
            return env, animation, algorithms
        if environment == "Bipedal Walker":
            env = BipedalWalker()
            animation = AnimationBipedalWalker(root, env, update_animation=True)
            algorithms = ["DDPG for Bipedal Walker"]
            return env, animation, algorithms