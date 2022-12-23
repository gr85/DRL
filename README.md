# DRL
Deep Reinforcement Learning repository to test different algorithms

## Using
The main file to run is _src/vecenv_test.py_. It is where the model and the training behvaiour are defined.

Before starting the code be sure that you have the same folder structure as this repository (_tmp_ and childs)

This project uses the _gymnasium_ package instead of _gym_, if you want to run the gymnasium package some changes in the _stable-baselines3_ package must be done in order to work properly with the HER buffer, because in the moment of writing it only supports gym package (the main changes are in the import, where you must change _gym_ for _gymnasium_). If you want to use _gym_ look inside my code and comment the parts where the note '''Panda-Gym V3''' is present and uncomment the lines with '''Panda-Gym V2''', then be sure you have installed _panda-gym==2.0.3_, because it is the verion which is using _gym_ yet.


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
