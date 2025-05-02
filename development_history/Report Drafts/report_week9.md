---
title: "Optimizing Decision-Making in Multi-Agent RL with Cumulative Prospect Theory"
author: "Sheyan Lalmohammed, Khush Gupta, Alok Shah"
---

# Optimizing Decision-Making in Multi-Agent RL with Cumulative Prospect Theory

**Authors:** Sheyan Lalmohammed, Khush Gupta, Alok Shah

---

## Abstract

This report details our work on incorporating Cumulative Prospect Theory (CPT) into Multi-Agent Reinforcement Learning (MARL) to better align agent behavior with human decision-making biases. Building on the theoretical foundations of CPT—namely loss aversion, reference-dependent evaluation, and probability weighting—we adapt a policy gradient framework (MADDPG) to train agents in competitive and cooperative environments. Our experiments in PettingZoo’s Simple Tag (competitive) and Simple Spread (cooperative) environments reveal distinct behavioral changes when varying the degree of CPT effects. We present quantitative reward trends and qualitative observations that demonstrate both risk-seeking and risk-averse agent behavior. Finally, we discuss our implementation strategy, key challenges, and next steps toward refining CPT integration in MARL settings.

---

## 1. Introduction

The goal of our work is to bridge the gap between conventional MARL—where agents maximize expected cumulative rewards—and human-like decision-making, which often deviates from classical rationality. By integrating CPT into the reinforcement learning framework, we investigate several core questions:

- **Adherence to CPT Functions:** Do CPT-trained agents follow the prescribed utility functions and probability distortions?
- **Strategy Optimization:** How do agents with CPT-based rewards optimize their strategies compared to those using standard expected reward maximization?
- **Emergent Dynamics:** In mixed populations, how do agents adjust their behavior based on the risk preferences of others?

Through our experiments, we seek to understand whether CPT effects can induce distinct risk-sensitive behaviors in competitive and cooperative multi-agent settings.

---

## 2. Background

### 2.1 Cumulative Prospect Theory (CPT)

Originally developed by Kahneman and Tversky (1979), Prospect Theory explains decision-making under risk by introducing three key concepts:
- **Loss Aversion:** Losses impact decision-making more significantly than equivalent gains.
- **Reference-Dependence:** Outcomes are evaluated relative to a reference point rather than in absolute terms.
- **Probability Weighting:** Individuals tend to overweight small probabilities and underweight large ones.

CPT extends this framework by integrating these behavioral insights into a model that can handle multiple outcomes. The value function is typically concave for gains and convex for losses, while the probability weighting function distorts the perceived likelihood of events.

### 2.2 Multi-Agent Reinforcement Learning (MARL)

In the MARL framework, multiple agents interact within a shared environment (e.g., Markov games). Each agent’s objective is to maximize its own expected return while adapting to the actions of others. The standard approach involves policy gradient methods that directly optimize the policy. However, the incorporation of CPT introduces nonlinearity—making the optimization landscape nonconvex and challenging to solve using traditional dynamic programming.

---

## 3. Implementation Strategy and Technical Approach

### 3.1 CPT-Adjusted Rewards and Policy Gradient

To incorporate CPT into MARL, we transform the standard reward signal using CPT’s value and probability weighting functions. Formally, the CPT-adjusted reward is expressed as:

$$ C(X) = \int_{-\infty}^0 w^+\bigl(P(u(X) > z)\bigr)\,dz - \int_{0}^{\infty} w^-\bigl(P(u(X) > z)\bigr)dz$$

and the CPT-based policy optimization problem becomes:

$$\max_{\pi \in \Pi_{M,N}} C\Bigl(\sum_{t=0}^{H-1} r_t\Bigr)$$

Our optimization strategy adapts the policy gradient theorem. Specifically, we compute:

$$\nabla_{\rho} J = \mathbb{E}\Bigl[\xi \Bigl(\sum_{\tau} r_{\tau}\Bigr), \nabla_{\rho},\mu_{\tau}\bigl(a_{\tau}\mid Q_{\tau}(s_{\tau},a_{\tau};n)\bigr),
  \nabla_{a_{\tau}},Q_{\tau}\bigl(s_{\tau},a_{\tau};n\bigr)
\Bigr]$$

with

$$\xi(V) =\ \int_{0}^{\max(V,0)} w^+\Bigl(P\bigl(u\bigl(\sum_{\tau} r_{\tau}\bigr) > z\bigr)\Bigr) dz - \int_{0}^{\max(-V,0)} w^-\Bigl(P\bigl(u\bigl(\sum_{\tau} r_{\tau}\bigr) > z\bigr)\Bigr)dz$$

This formulation, embedded within the Multi Agent Deep Deterministic Policy Gradient (MADDPG) framework, leverages model-free learning and automatic differentiation using PyTorch.

### 3.2 PyTorch Implementation Workflow

1. **Policy Network Architecture:**  
   Neural networks parameterize each agent’s policy.

2. **Reward Transformation:**  
   Traditional rewards are converted using CPT functions, capturing risk sensitivity via value distortions and probability weighting.

3. **Policy Gradient Computation:**  
   Gradients are computed with respect to the CPT-adjusted objective, followed by updates using gradient ascent.

4. **Multi-Agent Coordination:**  
   Agents share critical information during training to stabilize learning in both competitive and cooperative scenarios.

---

## 4. Experimental Evaluation

Our experiments are conducted in two distinct PettingZoo environments to assess the impact of CPT on agent behavior.

### 4.1 Competitive Environment: Simple Tag

**Setup:**  
In the Simple Tag environment, predators aim to catch a prey, with rewards structured to incentivize successful tagging. We compare three variants:
- **Baseline:** Standard expected reward maximization.
- **Moderate CPT (Risk Seeking):** Incorporating CPT adjustments with parameters that induce risk-seeking behavior.
- **Extreme CPT (Risk Averse):** Using parameters that significantly heighten loss aversion and risk aversion.

**Results:**  
The reward trends, as depicted in the figures, show that:
- Under **Moderate CPT**, predators and prey exhibit more exploratory, risk-seeking behavior.
- In the **Extreme CPT** scenario, agents tend to adopt conservative strategies, emphasizing safety over potential high gains.
  
*Figure comparisons (small thumbnails in the presentation) indicate clear shifts in the reward structure, aligning with our CPT parameterizations.*

### 4.2 Cooperative Environment: Simple Spread

**Setup:**  
In the Simple Spread environment, agents work cooperatively to cover landmarks while avoiding collisions. We again compare three settings:
- **Baseline:** Standard cooperative policy.
- **Moderate CPT:** Incorporates CPT adjustments to encourage moderate risk sensitivity.
- **Extreme CPT:** Applies strong CPT effects, leading to highly conservative (or in some cases, overly cautious) cooperative behaviors.

**Results:**  
Visualizations from the cooperative environment illustrate:
- **Baseline policies** yield coordinated movement with balanced risk-taking.
- **Moderate CPT policies** show subtle deviations, with agents adapting positions more dynamically to optimize coverage.
- **Extreme CPT policies** sometimes result in overly cautious behavior, impacting overall landmark coverage efficiency.

A linked visualization of the MPE (Multi-Agent Particle Environment) further confirms that while CPT integration alters movement patterns, it also introduces a spectrum of behaviors—from risk-seeking to risk-averse—depending on the parameter settings.

---

## 5. Analysis of Results

Our experiments demonstrate that integrating CPT into MARL has a measurable impact on agent behavior:

- **Behavioral Shifts:**  
  The degree of CPT adjustment (moderate vs. extreme) directly correlates with changes in agents’ risk profiles. In competitive settings, risk-seeking agents explore aggressive strategies, while risk-averse settings encourage safer, more predictable tactics.

- **Reward Dynamics:**  
  The CPT transformation introduces nonlinearity in the reward signal. This is evident in the distinct reward trends observed across experiments, with CPT agents deviating from classical Nash-equilibrium strategies.

- **Coordination and Adaptation:**  
  In cooperative scenarios, CPT-driven policies lead to adaptive behaviors where agents adjust their positions based on both their own risk preferences and those of their peers. However, extreme CPT parameters may dampen overall coordination efficiency.


## **Experiments Expanded**

In order to rigorously verify and understand CPT-based behavior in our multi-agent settings, we propose a series of experiments that evaluate both the qualitative and quantitative impacts of CPT transformations on agents’ policies and emergent dynamics.


### **1. Baseline Comparison: CPT vs. Traditional MARL Agents**

**Objective:**  
Determine if and how CPT-driven agents diverge from agents using standard expected reward maximization.

**Setup:**
- **Environments:** Use the predator-prey and Simple Tag environments from PettingZoo.
- **Agent Variants:**  
  - **CPT Agents:** Incorporate CPT-based value transformations and probability weighting functions.
  - **Standard Agents:** Use conventional RL objectives (e.g., expected cumulative rewards).

**Metrics:**  
- **Nash-Convexity Gap:** Compare how far each population deviates from classical Nash equilibria.
- **Divergence Metrics:** Compute Jensen-Shannon Divergence (JSD) and KL-Divergence between predicted action distributions and those expected under human-like CPT distortions.
- **Reward Dynamics:** Plot cumulative reward trends over training iterations.

**Expected Outcome:**  
CPT agents should display distinct risk-sensitive strategies—evidenced by different equilibrium properties and divergence in action selection compared to standard agents.


### **2. CPT Parameter Sensitivity Analysis**

**Objective:**  
Assess how varying the parameters in the CPT framework (e.g., $\alpha$, $\lambda$, $\beta$) affects agents' risk behavior and decision-making.

**Setup:**
- **Parameter Sweep:** Run experiments by varying:
  - **Risk Sensitivity:** $\alpha$ in the value function.
  - **Loss Aversion:** $\lambda$, controlling the penalty for losses.
  - **Probability Distortion:** $\beta$, affecting the weighting of probabilities.
- **Controlled Environment:** Use simplified scenarios with known risk/reward profiles to isolate the effect of each parameter.

**Metrics:**  
- **Action Distribution Shifts:** Monitor changes in agents’ preference for “safe” versus “risky” actions.
- **Performance Metrics:** Evaluate cumulative returns, Nash-Convexity Gap, and divergence measures as parameters change.

**Expected Outcome:**  
Parameter variations should lead to predictable changes in risk attitudes. For example, higher $\lambda$ is expected to increase loss aversion, while variations in $\beta$ should modify the degree to which low-probability events are overweighted.


### **3. Controlled Risk-Scenario Experiments**

**Objective:**  
Validate that CPT agents make decisions consistent with the CPT framework by setting up scenarios where risk profiles are explicitly defined.

**Setup:**
- **Scenario Design:** Create mini-games or decision nodes where agents choose between:
  - **A Safe Option:** Lower variance but moderate reward.
  - **A Risky Option:** Higher variance with potential for both high reward and steep losses.
- **Agent Variants:** Compare choices made by CPT-driven agents against those from standard agents.

**Metrics:**  
- **Choice Frequencies:** Record the proportion of times each option is selected.
- **Probability Weighting:** Analyze if the choice patterns align with the CPT probability weighting function (e.g., overweighting rare high-reward outcomes).

**Expected Outcome:**  
CPT agents should prefer options in a way that reflects diminishing sensitivity and loss aversion, diverging from the risk-neutral strategies of traditional agents.


### **4. Emergent Dynamics in Mixed-Agent Populations**

**Objective:**  
Study how agents with heterogeneous risk preferences interact and influence overall game dynamics.

**Setup:**
- **Mixed Populations:** Deploy environments containing a mix of:
  - Pure CPT agents (with varying parameter settings).
  - Traditional expected reward agents.
- **Dynamic Interaction:** Allow agents to interact over multiple episodes in competitive, cooperative, and mixed-motive scenarios.

**Metrics:**  
- **Adaptation and Coordination:** Track changes in policy coordination, strategy adaptation, and overall performance.
- **Equilibrium Analysis:** Evaluate the stability of learned strategies using Nash-Convexity Gap and divergence measures.
- **Information Elicitation:** Monitor if agents begin to strategically reveal or hide aspects of their risk preferences.

**Expected Outcome:**  
We expect to observe emergent dynamics where CPT agents display risk-sensitive behaviors that influence both cooperative and adversarial interactions. Their presence may also alter the equilibrium properties of the overall system, compared to homogeneous populations of standard agents.


### **5. Robustness to Environmental Noise**

**Objective:**  
Investigate whether CPT-based decision-making remains robust under varying levels of stochasticity and ambiguous reward landscapes.

**Setup:**
- **Noise Injection:** Introduce controlled noise into the reward signals or state transitions.
- **Environmental Variants:** Test across several environment configurations (e.g., different obstacle densities in the predator-prey setting).

**Metrics:**  
- **Policy Stability:** Assess how robust the CPT-driven policy is to noise by monitoring training stability and convergence rates.
- **Behavioral Consistency:** Compare the consistency of risk-sensitive decisions under noisy conditions versus noiseless conditions.

**Expected Outcome:**  
CPT agents should demonstrate resilience to noise, with their probability weighting and value distortions allowing them to better navigate uncertain outcomes compared to agents solely optimizing expected returns.


### **6. Gradient and Learning Dynamics Analysis**

**Objective:**  
Analyze how the introduction of CPT transformations affects policy gradient estimates and overall learning dynamics.

**Setup:**
- **Gradient Monitoring:** Track gradient norms and variance during training for both CPT and traditional agents.
- **Learning Rate Adjustments:** Experiment with different learning rate schedules to mitigate potential instability caused by nonconvex CPT objectives.

**Metrics:**  
- **Gradient Stability:** Record and compare gradient magnitudes and fluctuations.
- **Convergence Behavior:** Evaluate the number of iterations to reach certain performance thresholds, and analyze the smoothness of the reward progression curves.

**Expected Outcome:**  
Identifying potential challenges in gradient stability will inform further algorithmic adjustments (e.g., adaptive learning rates or regularization techniques) needed to successfully optimize CPT-driven policies.



---

## 6. Challenges and Future Directions

### 6.1 Current Challenges

- **CPT Integration Complexity:**  
  Implementing CPT-based transformations without destabilizing policy gradients remains a technical challenge. The nonconvex nature of the CPT objective requires careful tuning of learning rates and regularization methods.

- **Gradient Stability:**  
  The additional nonlinearity from CPT functions can result in unstable gradient updates. Ensuring smooth convergence necessitates further experimentation with adaptive optimization techniques.

- **Computational Overhead:**  
  Incorporating CPT adjustments increases the computational cost per training iteration. Optimizing these computations will be essential as we scale to more complex environments.

### 6.2 Future Work

- **Optimizing CPT Estimation:**  
  We plan to experiment with alternative estimation methods for value functions and integrals, potentially incorporating Monte Carlo rollouts or importance sampling strategies.

- **Exploring Discrete Competitive Domains:**  
  Extending the framework to environments like Poker could yield additional insights into interpretable CPT effects driven by behavioral economics.

- **Enhanced Multi-Agent Coordination:**  
  Future work will examine advanced coordination mechanisms to better handle heterogeneous risk preferences in mixed-agent populations.

---


## Next Steps

### Immediate Improvements Needed
- **Refine CPT Integration:** Further optimize the implementation of CPT-based reward transformations and probability weighting functions to ensure stable policy gradients.
- **Hyperparameter Tuning:** Fine-tune learning rates, regularization methods, and CPT parameters (e.g., \(\alpha\), \(\lambda\), \(\beta\)) to balance risk-seeking and risk-averse behaviors.
- **Enhance Computational Efficiency:** Optimize the additional computations introduced by CPT to minimize the impact on training time, especially in larger environments.

### Technical Challenges
- **Nonconvex Optimization:** The CPT objective introduces significant nonconvexities due to its nonlinear probability weighting and value transformations, complicating convergence.
- **Gradient Stability:** Managing the potential instability in gradient updates resulting from CPT distortions is critical, and may require adaptive optimization techniques.
- **Integration with Existing Frameworks:** Ensuring that CPT modifications do not destabilize established MARL algorithms like MADDPG while preserving their convergence properties.

### Anticipated Challenges
- **Multi-Agent Coordination:** Handling heterogeneous risk preferences among agents in mixed populations may lead to coordination issues or unexpected emergent behaviors.
- **Scaling to Complex Environments:** As the environments become more complex (e.g., discrete domains like Poker), integrating CPT in a robust manner will likely pose additional challenges.
- **Computational Overhead:** Increased computational demands from CPT adjustments could impact training efficiency, necessitating further optimization of the estimation methods.

### Further Questions
- How can CPT probability weighting functions be adapted effectively for continuous versus discrete action spaces?
- What are the most effective methods (e.g., Monte Carlo rollouts, importance sampling) for approximating CPT-weighted returns in complex environments?
- How will the integration of CPT affect the stability and convergence of learned equilibria in mixed-agent systems?
- Can additional coordination mechanisms be developed to mitigate adverse effects of heterogeneous risk preferences among agents?

### What You've Learned So Far
- **Baseline MARL Success:** Standard MARL approaches (e.g., MADDPG) work well in our environments, but they lack human-like decision-making biases.
- **Behavioral Shifts with CPT:** Incorporating CPT induces measurable changes in agent behavior, leading to a spectrum of risk-sensitive strategies ranging from risk-seeking to risk-averse.
- **Implementation Complexity:** Integrating CPT requires significant modifications to traditional reward and policy gradient formulations, emphasizing the need for careful hyperparameter tuning.
- **Preliminary Observations:** Initial experiments suggest that while CPT-based adjustments can enhance agent adaptability, they also introduce challenges related to training stability and computational efficiency.



## 7. Conclusion

Our work demonstrates the feasibility and impact of incorporating CPT into MARL. By adapting policy gradient methods to accommodate CPT’s probability weighting and value distortions, we have observed clear behavioral shifts in both competitive and cooperative environments. While challenges remain—particularly in gradient stability and computational efficiency—the preliminary results underscore the potential for CPT-based models to better align autonomous agents with human-like decision-making under risk.