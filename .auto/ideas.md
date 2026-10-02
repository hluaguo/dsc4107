# Ideas backlog

- A mean objective with an explicit tail penalty: `-mean + lambda * CVaR`,
  swept over `lambda`. This is the obvious bridge between the two ends of the
  objective sweep already recorded in the prompt.
- Differentiable Sharpe objective: `-mean / std` of the P&L, which aligns
  training with the primary metric directly instead of hoping the mean
  objective transfers.
- State features the current network ignores: realised volatility so far,
  time since the last trade, and the running hedge error. These are observable
  at the decision point and are exactly what a desk would condition on.
- A trade gate. Output a target position and a trade probability, then train
  with a straight-through or relaxed Bernoulli so the network can choose to
  hold still, which is how band strategies win under costs.
- Compare a recurrent policy against the feed-forward one. The current state is
  Markov enough in theory but a GRU may learn band behaviour faster.
- Learning rate schedule (cosine) and early stopping on the validation proxy,
  though early stopping must never read the holdout pool.
