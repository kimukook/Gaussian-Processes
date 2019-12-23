# Gaussian-Processes
This repo implements the Gaussian-Processes to fulfill the prediction task on regression problems.
In this repo, the function evaluations contain no noisy terms. The acquistion function implemented in this repo is GP-UCB. For each mesh refinement iteration, the coefficient controlling the amount of uncertainty will be doubled.

## How to use
Copy the code into the root folder, in terminal run the command
```
python3 1Dexample.py
```

## Results
At the begining, the algorithm still has a bunch of uncertainty around regions that are not evaluated.
![1](/figures/confidence_bound_earlystage.png)
At later stage, the uncertainty is reduced and the points cluster around the basin contains the maximizer.
![2](/figures/confidence_bound.png)


## References
[1]Rasmussen CE and Williams CK (2006) Gaussian processes for machine learning. Cambridge MA: MIT Press.

[2]Srinivas N, Krause A, Kakade SM and Seeger M (2012) Gaussian process optimization in the bandit setting: No regret and experimental design. IEEE Transactions on Information Theory 58(5): 3250–3265.
