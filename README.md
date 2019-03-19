[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Collaborative RL based on DDPG

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.


### Project details

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket (we are provided with 3 consecutive slices in time). Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### How to use this package
These instructions should help you navigate across project and possibly train you own agent or just evaluate already prepared models.

  1) Setup proper environment, more details in [install section](#install)   
  2) Explore structure of package (where are pretrained models, where is precompiled Tennis environment), more details in [structure section](#structure)  
  3) Check that environment is running and get to know with it, ```notebooks/environment_introduction.ipynb``` should help  
  4) Try to train your own agent, don't forget to setup your own path for saving model weights, ```notebooks/train.ipynb``` should help  
  5) In case you want to see results of already trained successful agents with description of method details, check ```notebooks/report.ipynb```

### <a name="install"></a> Install
 - ```python setup.py develop``` - This installs package with all requirements in develop mode for interactive updates.
 - ```install ``` [```Unity ML-Agents```](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) - Unity plugin that enables games and simulations to serve as environments for training intelligent agents, Unity environment is already compiled in project.
 - ```python setup.py test -a '-v  tests/'``` - Running all tests.

### <a name="structure"></a> Structure of package

 - ```pddpg_tennis``` - Package core with implementation of agent, environment, replay buffers etc.
 - ```tests```
 - ```bin``` - Compiled Unity environment ready for Linux operating systems.
 - ```experiments``` - Basic experiment **including trained 2 ddpg based actors and critic weights** prepared for pytorch + metadata files holding details about training. Details can be find in ```notebooks/report.ipynb```.
 - ```notebooks``` - Insight to usage of package and description of results.
   - ```train.ipynb``` - Example of agent training.
   - ```report.ipynb``` - Analysis of experiments, used architecture and proposal of next steps.
   - ```environment_introduction.ipynb``` - Example of wrapped environment usage.   
