# 2048-rl

Deep Q-Learning Project to play 2048.
See [this presentation](https://docs.google.com/presentation/d/1I9RS3SMdMp8Uk9C6eyS6jK_w_34BKCrvkN-kWau1MU4/edit?usp=sharing) for an introduction.

## Getting Started

Install [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/index.html), python & pip.
Then, run:

```bash
pip install -r requirements.txt
```

To run the code, you'll need to update your `PYTHONPATH`:

```bash
source set_pythonpath.sh
```

Now, you should be able to run the tests:

```bash
py.test
```

## Source Code Structure

All python source code lives in `py_2048_rl`.

### game

This directory contains code to simulate the 2048 game itself.
For example, it provides a `Game` class that implements the game logic.
The `play` module defines the `Experience` class, a `play()` function and various strategies that can be passed as an argument to `play()`.

### learning

This directory contains all code that has to do with the Deep Q-Learning algorithm itself.
Here's a comprehensive list of the modules:

- `replay_memory` implements the Replay Memory. Main methods are `add()` to add an experience and `sample()` to sample a number of experiences.
- `experience_collector` implements a `collect(strategy, num_games)` function that plays a number of games, deduplicates & undersamples the experiences, and returns them.
- `target_batch_computer` is responsible for computing the target batch that is passed to the network.
- `experience_batcher` uses the `ReplayMemory`, `ExperienceCollector` and `TargetBatchComputer` to generate training batches for the neural network.
- `model` defines the Neural Network architecture and its training parameters (e.g. Learning Rate).
- `learning` glues everything together to implement the Deep Q-Learning algorithm.

## Run Training

Step 1 is to set various parameters.

For example, you might want to adjust
- The `GAMMA` value in `target_value_computer.py`
- The `INIT_LEARNING_RATE` or `HIDDEN_SIZES` in `model.py`
- The `MIN_EPSILON` in `experience_batcher.py`
- ...

Once that's done, you can simple run `python py_2048_rl/learning.py <train_dir>`.

## Analyzing the Model

You can use TensorBoard to monitor your Network training, simply by passing you train directory as the `--logdir` param.
Furthermore, have a look at `py_2048_rl/analisis.py` (for plotting a historgram of Q-Values) and `py_2048_rl/play_game.py` (for simulating a (number of) games given a particular model).
