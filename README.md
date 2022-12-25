# DRL
The aim of this repository is to build different policy base algorithms (such as DDPG, TD3) of Deep Reinforcement Learning and test them in different environments.

This is part of the Final Master's Degree Thesis of the Master in Data Science from UOC (_Universitat Oberta de Catalunya_). Where we wanted to apply the state-of-the-art algorithms to deal with continuous action spaces, specially focused on solving some industrial robotics task. The main goal of this project was to solve a simple Pick and Place task involving an industrial robot.

To help the tests of these tasks we used a simulator based on PyBullet open source physics engine and designed by the team cited at the end.

The code which we developped used a HER buffer and a vectorized environments, for this reason it only works with vectorized environments. The implementation of HER were took from https://github.com/qgallouedec/stable-baselines3/blob/684364beddc53d206db38770db222aad1c599282/stable_baselines3/her/her_replay_buffer.py and adapted to work with gymnasium package.

The environments used must be the type of GoalEnv to work, because of the use of HER which is a mandatory condition.

## Using
The main file to run is _src/vecenv_test.py_. It is where the model and the training behaviour are defined.

Before starting the code be sure that you have the same folder structure as this repository (_tmp_ and childs)

This project uses the _gymnasium_ package instead of _gym_, if you want to run the gymnasium package some changes in the _stable-baselines3_ package must be done in order to work properly with the HER buffer, because in the moment of writing it only supports gym package (the main changes are in the import, where you must change _gym_ for _gymnasium_). If you want to use _gym_ look inside the code provided here and comment the parts where the note '''Panda-Gym V3''' is present and uncomment the lines with '''Panda-Gym V2''', then be sure you have installed _panda-gym==2.0.3_, because it is the verion which is using _gym_ yet.


## Citation
This work is based on the simulator developed by:

```bib
@article{gallouedec2021pandagym,
  title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
  year         = 2021,
  journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}
```
