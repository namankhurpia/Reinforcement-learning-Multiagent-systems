# Reinforcement-learning-Multiagent-systems
This repository contains multiagent reinforcement learning code implements Q learning
This is a grid world implementation of the famous mario game. This is basically a multi agent RL environment, where both the agents have a goal to achieve.
The agents are  -
  Mario
  Luigi
The rewards will be 
  Ghost  : -10
  Turtle   : -20
  Jumping point: -50
  Princess : +100
Policy :
  Either the ghost will surround the jumping point or the Turtle will surround the jumping point   
  The aim of Mario would be to reach the princess,avoid turtles, Ghosts, and Jumping points.
  The aim of Luigui would be to kill the princess, avoid turtles, Ghosts, and Jumping points.

Target / Stopping Point:
The game will end when Mario reaches the princess with a maximum reward.
Though the primary goal of the game is for Mario to reach the princess, we also keep track of the rewards that Mario has accumulated along the way. We will try to finish the game and tune it in such a way that Mario gets the maximum reward. We will be using 4 Q tables to choose actions, 2 for Mario and 2 for Luigui.
