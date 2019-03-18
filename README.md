[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Continuous Control with DDPG algorithm

### Introduction

In this project, we have trained robotic arm to follow concrete target via actor critic approach, specifically using [ddpg algorithm](https://arxiv.org/pdf/1509.02971.pdf). For experimental purposes, we have used [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment enabling continuous control.   

Apart from Python package with ddpg agent and training framework, we provide also already trained model with report addressing details of experiments.

![Trained Agent][image1]

### Project details

In Reacher environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with 4 numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the environment

#### Option 1: Solve Single Agent Environment

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve Multiagent Environment

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores.
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### How to use this package
These instructions should help you navigate across project and possibly train you own agent or just evaluate already prepared models.

  1) Setup proper environment, more details in [install section](#install)   
  2) Explore structure of package (where are pretrained models, where is precompiled Reacher environment), more details in [structure section](#structure)  
  3) Check that environment is running and get to know with it, ```notebooks/environment_introduction.ipynb``` should help  
  4) Try to train your own agent, don't forget to setup your own path for saving model weights, ```notebooks/train.ipynb``` should help  
  5) In case you want to see results of already trained successful agents with description of method details, check ```notebooks/report.ipynb```

### <a name="install"></a> Install
 - ```python setup.py develop``` - This installs package with all requirements in develop mode for interactive updates.
 - ```install ``` [```Unity ML-Agents```](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) - Unity plugin that enables games and simulations to serve as environments for training intelligent agents, Unity environment is already compiled in project.
 - ```python setup.py test -a '-v  tests/'``` - Running all tests.

### <a name="structure"></a> Structure of package

 - ```ac_continuous_control``` - Package core with implementation of agent, environment, replay buffers etc.
 - ```tests```
 - ```bin``` - Compiled Unity environment ready for Linux operating systems. It contain 2 alternatives, one for single agent (```bin/reacher_single```), one for multiple agents (```bin/reacher_multi```).
 - ```experiments``` - Basic experiment **including trained ddpg actor and critic weights** prepared for pytorch + metada files holding details about training. Details can be find in ```notebooks/report.ipynb```.
 - ```notebooks``` - Insight to usage of package and description of results.
   - ```train.ipynb``` - Example of agent training.
   - ```report.ipynb``` - Analysis of experiments and proposal of next steps.
   - ```environment_introduction.ipynb``` - Example of wrapped environment usage.   
