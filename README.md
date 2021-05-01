# TUM - Advanced Deep Learning for Robotics
This repository contains the project source code of our team ([@rajk853](https://github.com/rajk853), [@saif61](https://github.com/saif61)) for the TUM - Advanced Deep Learning for Robotics SS21 course.

## Objective
In this project, we will investigate the Reinforcement Learning (RL) approach in the Neural Motion Planning (NMP) as proposed in [T. Jurgenson and A. Tamar, 2019](https://arxiv.org/abs/1906.00214). We will compare the results between the Supervised Learning (SL) and Reinforcement Learning approaches and tweak the expert demonstration trajectories used in the Supervised Learning setting in an attempt to produce results similar to that from the Reinforcement Learning setting 

## Setup
1. Install [Conda](https://docs.anaconda.com/anaconda/install/linux/)
2. Clone this repository
```shell
git clone https://github.com/RajK853/tum-adlr-ss21-11.git .
```
3. Create and activate conda environment with following command  
```shell
cd tum-adlr-ss21-11
conda env create -f environment.yaml
conda activate adlr
```

## Usage

### Demo plot
> Available as an [notebook](\notebook\Demo_plot.ipynb).

Execute the given command where `${PATH_TO_DB_FILE}` is the location of the `.db` file in your local machine.
```shell
python demo_plot.py ${PATH_TO_DB_FILE}
```

## TODOs
- Setups for the Supervised Learning methods
  - Implement a `DenseNet` for the Image-to-Image path planning
  - Implement another model for the Image-to-Points path planning
- Setups for the Reinforcement Learning methods
  - Create a goal-based Gym-compliant RL environment  
  - Implement different variants of `Deep Deterministic Policy Gradient` (DDPG) algorithm
    - DDPG
    - DDPG+HER (Hindsight Experience Replay)
    - DDPG-MP (Motion Planning)
- Ideas to improve the Supervised Learning method results 
