# Animated Reinforcement Learning
My MSci Project which animates an agent running in various environments, using various reinforcement learning algorithms (including Deep RL and OpenAI gym environments)

#### Learning algorithms included:
* Value Iteration
* Policy Iteration
* First-visit Monte Carlo Control
* Q-Learning
* SARSA
* REINFORCE
* Actor-Critic
* Deep Q-Network
* Deep Deterministic Policy Gradient

#### Environments included:
* Frozen Lake (grid world)
* Cliff Walking (grid world)
* Taxi Driver (grid world)
* CartPole (OpenAI gym)
* Atari 2600 Pong (OpenAI gym)
* Lunar Lander (OpenAI gym)
* Bipedal Walker (OpenAI gym)

#### Other features:
* Configuration text file for adjusting all parameters used by the learning algorithms
* Animation speed slider
* Animation toggle
* Statistical information output during learning
* Graph displayed of average rewards per episode at the end of learning

## To Run
**OS:** Linux

**Python:** 3.7.4+

Open a directory in the root folder of the repository.

Install the package dependicies using pip:
```
pip install -r requirements.txt
```

Run the program:
```
chmod u+x run.sh
./run.sh
```

![screenshot](https://i.imgur.com/nrEl4Kg.png)

## Video Demos 
Deep Q-Network for CartPole Environment:

https://www.youtube.com/watch?v=eddZEhWoBUs&feature=youtu.be

Q-Learning for Taxi Driver Environment:

https://www.youtube.com/watch?v=EAe37d4bwUM


Value Iteration for Frozen Lake Environment:

https://www.youtube.com/watch?v=Q_KDa2du_mw
