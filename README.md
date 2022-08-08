# RNN tutorial

RNN tutorials for the ErUM-Data-Hub Deep Learning School https://indico.scc.kit.edu/event/2851/

We will use Exercises from Chapter 9 of http://deeplearningphysics.org and additional exercises from this repository:

1) Understand the `keras` RNN implementations: [understand_rnns.ipynb](understand_rnns.ipynb)
2) Predict a sine curve - [Exercise_09_1.ipynb](https://github.com/DeepLearningForPhysicsResearchBook/deep-learning-physics/blob/main/Exercise_09_1.ipynb) from *Deep Learning for Physics Research*
  ```python
  # Technical hint: Exercise_09_1 runs sufficiently fast on CPU
  # but it may be beneficial to restrict the number of cores in the first notebook cell via
  import tensorflow as tf
  tf.config.threading.set_intra_op_parallelism_threads(1)
  tf.config.threading.set_inter_op_parallelism_threads(1)
  ```
3) Identify cosmic ray signals - [Exercise_09_2.ipynb](https://github.com/DeepLearningForPhysicsResearchBook/deep-learning-physics/blob/main/Exercise_09_2.ipynb) from *Deep Learning for Physics Research* - recommended to run on GPU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLearningForPhysicsResearchBook/deep-learning-physics/blob/master/Exercise_09_2.ipynb)
4) Work with variable length sequences - [variable_length_masking.ipynb](variable_length_masking.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikoladze/deep-learning-rnn-tutorial/blob/master/variable_length_masking.ipynb)
