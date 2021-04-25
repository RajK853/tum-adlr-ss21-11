# TUM - Advanced Deep Learning for Robotics
This repository contains the project source code of our team ([@rajk853](https://github.com/rajk853), [@saif61](https://github.com/saif61)) for the TUM - Advanced Deep Learning for Robotics SS21 course.

## Objective
As part of this project, we will investigate the ideas proposed by [Li and Malik, 2016](https://arxiv.org/abs/1606.01885) for an optimization based motion planning using RL. 

## TODOs
- Set up an optimizer for a simple robot
  - Setups for a simple robot:
    - [Gym Minigrid](https://github.com/maximecb/gym-minigrid): Minimalist OpenAI Gym grid environments for path planning. 
  - Setups for the optimization algorithms: 
    - `gradient descent`: [Adam - tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
    - `momentum`: [SGD - tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
    - `conjugate gradient`: [Tensorflow 2.4.1](https://www.tensorflow.org/api_docs/python/tf/linalg/experimental/conjugate_gradient), [Numpy](https://gist.github.com/glederrey/7fe6e03bbc85c81ed60f3585eea2e073), [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)
    - `L-BFGS`: [Tensorflow Probability](https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize), [Scipy](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
- Implement RL to guide the optimizer i.e. `autonomous optimizer`

## Setup
...

## Usage
...

## Materials
- [Mathematical Optimizations](https://scipy-lectures.org/advanced/mathematical_optimization/)