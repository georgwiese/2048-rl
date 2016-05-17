# `train_0_3`

Trained from tag 0.3. Some key parameters:

- Start Learning rate is 1e-4.
- Network architecture is fully connected with hidden sizes [256, 256].
- Gamma is 0.
- Reward is -1 if lost, 0 otherwise
- --> Essentially it's a lost (-1) / not lost (0) classification

Undersampling parameters are such that the memory stats are:
```
Memory stats:
  Experiences:  100000
  Unavailable:  0 (0.0%)
  Lost       :  19139 (19.1%)
```
