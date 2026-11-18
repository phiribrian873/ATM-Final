# ATM Cash Replenishment Optimization Using Deep Q-Network Reinforcement Learning

## 1. Project Overview

This project explores the application of Deep Q-Network (DQN) reinforcement learning to ATM cash replenishment optimization.

The objective is to develop and evaluate an intelligent decision-making model designed to reduce ATM operating costs while maintaining reliable cash availability.

The DQN model is compared against two baseline approaches:

1. Cash Threshold Policy
2. Fixed Schedule Policy

The project evaluates operating costs, replenishment frequency, cash availability, unmet customer demand, and training stability.

## 2. Project Objectives

- Develop an ATM cash replenishment decision-making model using DQN reinforcement learning.
- Train the model using ATM cash management data and a simulated environment.
- Compare DQN performance against traditional replenishment strategies.
- Evaluate model performance using repeated testing.
- Assess deployment stability using predefined performance conditions.

## 3. Technologies Used

- Python
- Google Colab
- PyTorch
- Pandas
- Matplotlib
- GitHub
- Deep Q-Network Reinforcement Learning

## 4. Repository Files

| File | Description |
|---|---|
| ATM_Final.ipynb | Google Colab notebook containing result analysis and visualizations |
| atm_dqn_assignment.py | Python implementation of the ATM DQN project |
| atm_dqn.pt | Saved trained DQN model |
| training_log.csv | Training rewards across 350 episodes |
| test_results.csv | Results from repeated policy evaluations |
| test_summary.csv | Summary of DQN and baseline policy performance |

## 5. Running the Project

1. Open the GitHub repository.
2. Open `ATM_Final.ipynb` using Google Colab.
3. Connect to the Colab runtime.
4. Execute the notebook cells in sequence to reproduce the saved result analysis.

To execute the Python implementation separately, use:

```python
!python atm_dqn_assignment.py
```

Executing the Python script might repeat training or testing, depending on its implementation.

## 6. Experimental Results

The DQN model was trained for 350 episodes and evaluated across 10 test runs.

### Performance Comparison

| Performance Metric | DQN | Cash Threshold | Fixed Schedule |
|---|---:|---:|---:|
| Average operating cost | 11.3514 | 11.4675 | 12.3673 |
| Average replenishments | 7.0 | 7.3 | 8.0 |
| Cash availability (%) | 98.04 | 100.00 | 100.00 |
| Average unmet demand | 0.1808 | 0.0000 | 0.0000 |

The DQN achieved approximately 1.01% lower average operating costs than the Cash Threshold Policy and 8.21% lower costs than the Fixed Schedule Policy.

Despite these cost reductions, the DQN recorded lower cash availability and higher unmet demand.

## 7. Training Performance

The DQN model completed 350 training episodes.

The 20-episode moving-average reward was examined to assess training convergence.

### Training Stability Results

| Measurement | Result |
|---|---:|
| Average reward, episodes 251–300 | -8.8280 |
| Average reward, episodes 301–350 | -9.2287 |
| Final 20-episode moving-average reward | -9.6551 |

The results suggest the training reward approached a plateau during the later episodes.

## 8. Evaluation Stability

The trained DQN policy was evaluated across 10 test runs.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Operating cost | 11.3514 | 1.1927 |
| Cash availability (%) | 98.0357 | 1.0137 |
| Unmet demand | 0.1808 | 0.1194 |

The results demonstrate variation across test runs. A predefined variability threshold is required before confirming whether the model satisfies the low-variability condition.

## 9. Deployment Stability Assessment

Four conditions were considered when assessing deployment readiness.

| Requirement | Assessment |
|---|---|
| Moving-average reward plateau | Supported by training results |
| Low variability across evaluations | Not conclusively established |
| Cash availability at least equal to both baselines | Failed |
| Operating cost lower than both baselines | Passed |

### Deployment Decision

The DQN model demonstrated improved cost efficiency compared with both baseline policies.

Nevertheless, average cash availability was 98.04%, compared with 100% for both baseline policies.

The model therefore does not satisfy all deployment stability requirements.

**Final Decision: Not approved for deployment under the current evaluation criteria.**

## 10. Recommendations

Future improvements should focus on:

- Introducing stronger penalties for cash shortages during training.
- Adjusting the reward function to prioritize cash availability.
- Testing additional training configurations.
- Evaluating the model across more demand scenarios.
- Reassessing performance against the baseline policies after improvements.

## 11. Conclusion

This project demonstrates the application of Deep Q-Network reinforcement learning to ATM cash replenishment optimization.

The trained model achieved lower average operating costs and fewer replenishments than the baseline approaches.

The evaluation also identified limitations in cash availability and unmet demand.

Further model improvement and validation are necessary before operational deployment.
