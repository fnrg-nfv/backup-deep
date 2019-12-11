# How to train

- When has no improvement, adjust the LEARNING_RATE to a smaller value, and again;

# Result

- `start` and `end` in `state` have no influence of final result, other states of a sfc has a little influence(about 5%)；
- **LOW learning rate also matters**, when the learning rate is 1e-6, it also can achieve the same performance as a higher learning rate such as 1e-3(may be a little higher);
- In ICC algorithm mainly are two factors, one is the computing resource, the other is the length of update path, but due to experiments, the latter is more important.

# Debate on jitter or not jitter

#

# Debate on state transitions



