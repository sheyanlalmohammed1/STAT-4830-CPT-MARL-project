# Week 7 Report

## **Problem Statement**

### **Introduction**



We aim to optimize decision-making paradigms in Multi-Agent Reinforcement Learning (MARL) systems by perturbing both the value and distribution of rewards, guided by key insights from Cumulative Prospect Theory (CPT) (Kahneman & Tversky 1979).

CPT, a well-established behavioral economic framework, proposes the following key ideas:
- Individuals evaluate outcomes with respect to a reference point
rather than in absolute terms.
- Individuals are more sensitive to losses than gains (loss aversion).
- The value function is concave for gains and convex for losses, reflecting diminishing sensitivity.


Given recent advances reasoning and alignment (Guan 2024, DeepSeek-AI 2025), we find it essential to develop autonomous agents that *align with human preferences*. Prior literature suggests CPT provides an interpretable model for achieving this, even in settings characterized by noisy objectives, nonconvex reward landscapes, and ambiguous decision-making criteria (L.A 2016, Danis 2023, Lepel & Barakat 2024, Ethayarajh 2024).

Moreover, we propose MARL as an ideal testbed for evaluating the efficacy of our approach, offering a computationally tractable, scalable environment to study learned policies under prospect-theoretic reward structure (Zhang 2021). In particular, we seek to address the following research questions:

- Do CPT trained agents work follow their utility and probability distortion functions?

- How do CPT-guided agents optimize strategies in multi-agent games, and how do their behaviors differ from those using traditional utility functions?

- To what extent do agents adapt their strategies based on the utility functions of counterparties? What emergent dynamics arise in mixed populations of agents?

- Can agents strategically elicit information about their counterparties’ utility functions while minimizing the disclosure of their own preferences? What is the impact of this asymmetry on game outcomes?



## **Preliminaries**  

### **Multi-Agent Reinforcement Learning (MARL)**  

A **Markov Decision Process (MDP)** is defined as a tuple $M = (S, A, P, r, \rho, \gamma)$, where:  
- $S$ is the state space.  
- $A$ is the action space.  
- $P(s' | s, a)$ is the state transition probability.  
- $r(s, a)$ is the bounded reward function.  
- $\rho$ is the initial state distribution.  
- $\gamma \in (0,1)$ is the discount factor.  

A **policy** $\pi(a | s)$ defines a probability distribution over actions given a state. The objective in standard RL is to maximize the expected cumulative discounted return:  

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{H-1} \gamma^t r(s_t, a_t) \right].
$$

Extending this framework to the **multi-agent setting**, MARL considers \( N \) agents interacting within a shared environment, modeled as a **Markov game**:

$$
M = (N, S, \{A_i\}_{i=1}^{N}, P, \{r_i\}_{i=1}^{N}, \gamma).
$$

Each agent $ i $ selects actions from its individual action space $ A_i $ according to policy $\pi_i(a_i | s)$, and receives reward $r_i(s, a)$ based on the joint action $a = (a_1, ..., a_N)$. The goal of each agent is to optimize its own expected return, while accounting for interactions with other agents. MARL formulations include:  
- **Cooperative**: Agents share a common reward function.  
- **Competitive**: Agents maximize conflicting objectives.  
- **Mixed-Motive**: Agents exhibit both cooperative and adversarial behaviors.  

A key challenge in MARL is equilibrium computation. The **Nash equilibrium** in this setting is a strategy profile where no agent benefits from unilateral deviation. However, in the presence of **CPT-driven decision-making**, agents may deviate from classical Nash equilibria, instead optimizing for prospect-theoretic utilities.

### **CPT-RL**  

Traditional RL assumes that agents maximize **expected** rewards. CPT, in contrast, models decision-making under risk by incorporating **value distortions** and **probability weighting**. A CPT-driven agent evaluates returns as:

$$ \mathbb{E}_{\pi} \left[ w(P(v(G_t) > z)) \right] $$
where: 
- $G_t = \sum_{t=0}^{H-1} r(s_t, a_t)$ is the cumulative return.  
- $v(x)$ is the **value function**, which is concave for gains and convex for losses.  
- $w(p)$ is the **probability weighting function**, which distorts the perception of probabilities.  

**Value Function:** Kahneman and Tversky (1979) proposed the following formulation:

$$
v(x) =
\begin{cases}
    x^\alpha, & x \geq 0 \\
    -\lambda (-x)^\alpha, & x < 0
\end{cases}
$$

where $\lambda > 1$ represents **loss aversion**, and $\alpha \in (0,1)$ captures **diminishing sensitivity**.  

**Probability Weighting Function:** The distortion of probabilities is modeled as:

$$
w(p) = \frac{p^\beta}{(p^\beta + (1-p)^\beta)^{1/\beta}},
$$

where $\beta$ controls the overweighting of rare events and underweighting of likely events.  

By incorporating these distortions, CPT-RL agents deviate from classical **rational** behavior, making risk-sensitive decisions that resemble human biases.

### **Cumulative Prospect Theory (CPT) in Reinforcement Learning**  

Traditional RL agents maximize **expected** rewards. CPT, in contrast, models decision-making under risk by incorporating **value distortions** and **probability weighting**. A CPT-driven agent evaluates returns as:

$$
\mathbb{E}_{\pi} \left[ w(P(v(G_t) > z)) \right]
$$

where:
- $G_t = \sum_{t=0}^{H-1} r(s_t, a_t)$ is the cumulative return.  
- $v(x)$ is the **value function**, which is concave for gains and convex for losses.  
- $w(p)$ is the **probability weighting function**, which distorts the perception of probabilities.  

#### **Definition of the CPT Value Function \( C(X) \)**  

Following L.A. et al. [2016], the **CPT value** of a real-valued random variable $ X $ is given by:

$$
C(X) = \int_{0}^{\infty} w^+(P(u^+(X) > z))dz - \int_{0}^{\infty} w^-(P(u^-(X) > z))dz,
$$

where assuming appropriate integrability. While the CPT value $ C(X) $ accounts for **human agents' distortions in perception**, it also **recovers the expectation** $ E(X) $ when the weight functions are identity.


## **Optimization Problem**  


We seek to optimize agents' ability to find **non-stable equilibria** by learning a CPT-driven policy

Formally, we define the **CPT policy optimization (CPT-PO), ** objective as:

$$
\max_{\pi} C\left[ \sum_{t=0}^{H-1} r(s_t, a_t) \right],
$$

where $C(\cdot)$ is the CPT-value of cumulative returns.

Notably, CPT-value function does **not satisfy a Bellman equation** by nonlinearity of both the utility and probability weighting functions. This renders classical dynamic programming technqiues ineffective as additivity and linearity of the standard expected return formulation no longer exist.

Furthermore, the **CPT-PO** objective is nonconvex:
- The **utility function** is nonconvex in general (convex for losses, concave for gains).  
- The **probability weighting function** introduces additional nonlinear, nonconvex distortion
- The standard policy optimization problem is already nonconvex in the policy, and CPT-PO **adds further nonconvexity**, making traditional convergence guarantees difficult to establish.  


We hypothesize that CPT-trained agents will exhibit **risk-heterogeneous behaviors**, leading to emergent dynamics distinct from traditional MARL equilibria.


## **Evaluation Metrics**

We measure the performance of CPT-trained agents using the following key metrics:

1. **Nash-Convexity Gap**:  
   - Measures deviation from the classical Nash equilibrium.
   - Quantifies how far agents deviate from optimal cooperative or adversarial strategies.

2. **Jensen-Shannon Divergence (JSD) / Kullback-Leibler Divergence (KL-Div)**:  
   - Evaluates whether agents assess counterparties’ actions with the same probability as humans.
   - Provides insight into the degree of probability distortion in decision-making.

To further validate our model, we implement the **predator-prey MARL environment**, where:
- **Good agents** (prey) receive a negative reward for being captured.
- **Adversarial agents** (predators) receive positive rewards for capturing prey.
- **Obstacles** constrain movement, requiring strategic risk-sensitive decision-making.

Experiments will be conducted using **PettingZoo, Gym, and other premade MARL environments**, with moderate computational demands.

---


<!-- We are attempting to optimize the ability of multiple agents to find a non-stable equilibrium by learning a CPT-driven policy. This can effectively be measured by using a utility-focused method, such as Nash-Convexity gap, which measures the aggregate divergence of the agents from a pre-supposed classical equilibrium. Alternatively, we can use a probability weighting method, such as a Jensen-Shannon Divergence or Kullback-Leibler Divergence, to determine if the agent is assessing the actions of other agents with the same probability as would a human.

We assume that an agent operates under some set of constraints and aims to take the optimal action using heterogeneous risk preferences. To illustrate this, we consider the actions of multiple competitive agents in a predator-prey environment. Good agents are faster and receive a negative reward for being hit by adversaries. Adversaries are slower and are rewarded for hitting good agents. Ther are obstacles which block the way for both types of agents. We plan on continuing to use premade reinforcement learning environments (PettingZoo, Gym, etc.) to conduct all experiments and expect moderate use of compute. -->


## Technical Approach

We adopt a policy gradient approach tailored for CPT-based objectives, building upon the work of Lepel and Barakat (2024). Their research introduced a CPT-adjusted policy gradient theorem, enabling the design of a model-free policy gradient algorithm. This method is suitable for our multi-agent setting because:

- It allows for continuous action spaces and handles non-linearity introduced by CPT.
- Policy gradient methods are effective for high-dimensional problems and can achieve convergence under certain conditions.
- Unlike value-based methods (e.g., Q-learning), policy gradients directly optimize the policy, making them robust in multi-agent and stochastic environments.

### PyTorch Implementation Strategy

Our PyTorch implementation follows these key steps:

1. **Policy Network Architecture**  
   - Design neural networks to parameterize the policy $\pi_{\theta}$ for each agent, where $\theta$ represents network parameters.

2. **CPT-Based Reward Transformation**  
   - Implement CPT value and probability weighting functions to transform traditional rewards into CPT-adjusted rewards.

3. **Policy Gradient Computation**  
   - Compute the gradient of the CPT-objective with respect to \( \theta \) using automatic differentiation.

4. **Optimization**  
   - Update the policy parameters using gradient ascent with an appropriate optimizer

5. **Multi-Agent Coordination**  
   - Implement mechanisms for agents to share relevant information and coordinate during training, ensuring stability and convergence.

### Validation Methods

- **Simulation Environments**  
  - Currently, we are using Simple Tag from PettingZoo, a multi-agent reinforcement learning (MARL) environment designed for competitive behavior.
  - After establishing a working formulation, we may explore more complex multi-agent games and negotiation scenarios.

- **Nash-Convexity Gap Analysis**  
  - Measure how far agents deviate from a pre-supposed Nash equilibrium.

- **Divergence Metrics**  
  - Compute Jensen-Shannon Divergence (JSD) and Kullback-Leibler Divergence (KL-Div) to evaluate how closely the learned policy aligns with human-like CPT decision-making behaviors.

### Resource Requirements and Constraints

- **Compute Resources**  
  - Moderate GPU utilization expected for training reinforcement learning models.
  - Scalability concerns if multi-agent interactions require a large number of simultaneous agents.




## Initial Results

### Evidence Your Implementation Works
The implementation successfully trains agents in a **competitive multi-agent environment** using **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**. The plot generated in the notebook shows **reward trends over training iterations**, indicating that agents are learning effective strategies. Additionally, replay buffers and policy updates through `DDPGLoss` and `SoftUpdate` confirm that training dynamics are functioning as expected. **However, nothing related to Cumulative Prospect Theory (CPT) has been implemented yet.**

![](figures/Figure.png)

### Basic Performance Metrics
- **Reward Convergence:** The plotted training rewards illustrate that agents improve their performance over time, demonstrating that the learning process is effective.
- **Policy Optimization:** The use of `polyak_tau = 0.005` for soft updates stabilizes policy learning, ensuring that agents avoid drastic policy shifts.
- **Exploration Efficiency:** The training setup effectively balances exploration and exploitation, as observed in the **reward progression graph**.
- **CPT Implementation Status:** No modifications related to **Cumulative Prospect Theory (CPT)** have been incorporated yet.

### Test Case Results
- **Reward Trends:** The plot showcases increasing cumulative rewards, signifying that agents are **successfully adapting** to the environment.
- **Environment Validation:** The call to `base_env.full_reward_spec` confirms that reward structures are correctly implemented.
- **Performance Benchmarking:** While explicit win rates are not tracked, the observed learning curves imply effective training.
- **CPT Testing Pending:** No CPT-based decision-making mechanisms have been tested or validated yet.

### Current Limitations
- **Lack of Final Evaluation Metrics:** No explicit win/loss ratio or episodic score distributions are logged for post-training validation.
- **Computational Overhead:** The replay buffer (`1M` frames) demands significant memory, though its efficiency isn't directly measured.
- **Training Stability:** No explicit stopping criterion is defined for convergence assessment, making it unclear when optimal performance is reached.
- **No CPT Integration:** The current implementation does not include any CPT-based probability distortions or decision-making mechanisms.


## **Experiments**

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


## Next Steps

### Immediate Improvements Needed

- We have successfully created a multi-agent reinforcement learning (MARL) environment and trained agents using Deep Deterministic Policy Gradient (DDPG). However, we need to refine the training stability and reward shaping before integrating CPT.  
- Once we receive the codebase from the authors of the referenced paper, we will incorporate CPT-driven decision-making and evaluate its impact.  
- Further hyperparameter tuning is required to ensure convergence and minimize training instability introduced by multi-agent interactions.  

### Technical Challenges

- **CPT Integration**: Implementing CPT-based probability weighting and value transformation into an existing RL framework without destabilizing learning dynamics.  
- **Gradient Stability**: Since CPT alters the reward structure, we need to ensure stable policy updates and prevent vanishing/exploding gradients.  
- **Multi-Agent Coordination Under CPT**: Standard MARL algorithms assume rational agents; introducing CPT may create unexpected strategic deviations, requiring new coordination mechanisms.  
- **Computational Overhead**: Incorporating CPT increases memory and computation requirements, necessitating optimization strategies to ensure feasible training times.  

### Anticipated Challenges

Throughout this project, there are several challenges we anticipate:

1. **Convergence Difficulties**  
   Achieving convergence using a CPT-based policy function in reinforcement learning is likely to be challenging. Since CPT introduces non-linear probability distortions and reference dependence, standard learning algorithms may struggle to find stable solutions.

2. **Non-Convexity of the Optimization Landscape**  
   CPT modifies the reward function in ways that introduce local optima and gradient instability. This could lead to unstable training dynamics, requiring regularization techniques or adaptive learning rates.

4. **Multi-Agent Coordination**  
   Agents with heterogeneous risk preferences (due to different CPT parameters) may fail to coordinate effectively. This could lead to suboptimal collective behavior, particularly in collaborative or adversarial settings.

5. **Computational Complexity**  
   Applying CPT transformations at every step increases computational overhead. This is particularly problematic for large-scale simulations, requiring efficient approximation methods for probability weighting and decision modeling.


### Further Questions

- How should CPT probability weighting functions be adapted for continuous action spaces? Most existing implementations focus on discrete decision-making, making adaptation to DDPG-based continuous policies non-trivial.  
- What is the best strategy to approximate CPT-weighted returns? Should we use Monte Carlo rollouts, importance sampling, or direct function approximation?  
- How will CPT impact equilibrium stability in MARL? Will agents develop cyclic or unstable behaviors due to probability distortions, and how can we mitigate this?  

### What You've Learned So Far

- DDPG performs well in our MARL environment, but training stability remains a challenge, particularly in complex interactions.  
- Multi-agent learning introduces significant variability, making it difficult to evaluate policy effectiveness without careful tuning.  
- CPT introduces fundamentally different decision-making dynamics, and integrating it into MARL requires careful adjustments to reward functions, learning rates, and exploration-exploitation tradeoffs.  
- Waiting for the reference implementation is crucial, as it will provide insight into CPT policy gradient adaptations and help avoid redundant implementation work. 