# Modeling Human Behavior Without Humans -- Bringing Prospect Theory to Multi-Agent Reinforcement Learning

## Project Team Members

Sheyan Lalmohammed, Khush Gupta, Alok Shah

## High-Level Summary

- **Final Problem Statement:**  
  How can we endow multi-agent reinforcement learning (MARL) agents with human-like risk preferences—specifically loss aversion and probability weighting as characterized by Cumulative Prospect Theory—so that they exhibit calibrated risk-seeking or risk-averse behaviors in competitive, cooperative, and auction-based settings without human intervention?

- **Approach:**  
  We develop CPT-MADDPG, a differentiable extension of the MADDPG actor–critic framework that replaces expected-return objectives with CPT-based Choquet integrals over gains and losses. We approximate the CPT integral via batch‐based empirical tail‐probability estimates and piecewise‐linear probability weighting, integrate rank-dependent transforms directly into both critic and actor updates, and introduce two novel modules:  
  1. **Observability Adjustment**, which aggregates peers’ subjective utilities into the Bellman backup when agents share CPT parameters.  
  2. **Adaptive Behavioral Parameters**, which treat CPT hyperparameters (\(\alpha,\beta,\lambda\)) as learnable variables updated via a secondary loss.

- **Key Findings / Contributions:**  
  1. **Behavioral Modulation:** Moderate CPT hyperparameters accelerate exploration and early learning, while extreme parameters enforce conservative, low-variance strategies at a performance cost.  
  2. **Coordination under Transparency:** Allowing agents to observe each other’s CPT utilities preserves cooperative equilibria even with heterogeneous risk profiles.  
  3. **Limits of Adaptivity:** Naive dynamic adaptation of CPT parameters destabilizes learning and prevents convergence.  
  4. **Auction Overbidding:** CPT-trained bidders systematically overbid and replicate the short-term gain versus long-term loss


## Repository Structure

```
CPT-MARL/
├── README.md                    # This file
├── development_history          # All Previously Worked on Content for this Project
   ├── assignments/         # Moved from original location in docs (contains figures and old midterm presentation)
   ├── llm_exploration/           # Logs of LLM Use and Feedback from over the semester
   ├── old_code/                  # Old Code files from over the semester
   ├── report_drafts/             # Report Drafts from over the semester
   ├── self_critiques/            # Self-Critiques from over the semester
├── report.pdf                    # Our Final Report (to be submitted to a conference soon)
├── notebooks/                   # Final Jupyter Notebooks
   ├── CPT_MADDPG_Competitive.ipynb    # Naive Approximation to CPT Integral with Competitive Simple Tag Environment
   ├── CPT_MADDPG_Cooperative.ipynb    # Naive Approximation to CPT Integral with Cooperative Simple Spread Environment
   ├── CPT_MADDPG_QTE_Cooperative.ipynb          # Improved Approximation to CPT Integral with First Price Auction
   ├── CPT_MADDPG_First_Price_Auction.ipynb       # Improved Approximation to CPT Integral with Cooperative Simple Spread Environment
   ├── MADDPG_Cooperative_Adaptive_Behavioral_Parameters.ipynb        #Naive Approximation with Dynamically Changing CPT Hyperparameters
   ├── MADDPG_Cooperative_Allowing_Observe_Utility_Adjusted_Rewards.ipynb       #Naive Approximation with Agents Observing Utility Adjusted Rewards:
├── src/                        # Source code (Empty)
├── tests/                      # Test files (Empty)
└── docs/
   ├── presentation.pdf         # Final Presentation
```

# Setup Instructions

The setup for our code is relatively straightforward. You can simply run the code directly in colab. Note, it may take a while for each code file to run (we highly recommend using a GPU to do the coding) considering the fact that you have to run hundreds of iterations to see our results come to life. 

# Running the Code

You can replicate our experiments by using the hyperparameters detailed in the report and putting them into the notebooks for the respective experiment. Changing those values in the notebook will change the experiment being ran, repeating for all said experiments will result in the plots and values seen throughout our final report and final presentation. You can also directly create a visualization of the MPE environments within the code itself which saves locally. For reference, here are the hyperparameters to the CPT functions to replicate our results (the MPE environments use our original linear approximation to the CPT integral but the improved approximation is included for reference):

| Environment       | Variant                               | α    | β    | λ    | γ    | δ    | (w⁺)' | (w⁻)' |
| ----------------- | ------------------------------------- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| **Simple Tag**    | Baseline                              | N/A  | N/A  | N/A  | N/A  | N/A  | N/A   | N/A   |
| Simple Tag        | Moderate CPT (risk‐seeking)           | 0.90 | 0.60 | 1.50 | 0.69 | 0.61 | 0.80  | 0.20  |
| Simple Tag        | Extreme CPT (risk‐averse)             | 0.88 | 0.88 | 2.25 | 0.61 | 0.69 | 0.20  | 0.80  |
| **Simple Spread** | Baseline                              | N/A  | N/A  | N/A  | N/A  | N/A  | N/A   | N/A   |
| Simple Spread     | Moderate CPT (risk‐averse)            | 0.88 | 0.88 | 2.25 | 0.61 | 0.69 | 0.20  | 0.80  |
| Simple Spread     | Extreme CPT (risk‐averse)             | 0.70 | 0.95 | 2.50 | 0.61 | 0.69 | 0.20  | 0.80  |
| Simple Spread     | Observability CPT (Seeing – RS Agent) | 0.70 | 0.70 | 0.80 | 0.61 | 0.69 | 0.80  | 0.20  |
| Simple Spread     | Observability CPT (Seeing – RA Agent) | 0.65 | 0.65 | 2.80 | 0.61 | 0.69 | 0.25  | 0.75  |
| Simple Spread     | Dynamic (Agent 1)                     | 0.70 | 0.70 | 2.50 | 0.61 | 0.69 | 0.80  | 0.20  |
| Simple Spread     | Dynamic (Agent 2)                     | 0.65 | 0.65 | 2.80 | 0.61 | 0.69 | 0.80  | 0.20  |
| Simple Spread     | Dynamic Moderate (Agent 1)            | 0.60 | 0.60 | 1.00 | 0.50 | 0.55 | 0.20  | 0.80  |
| Simple Spread     | Dynamic Moderate (Agent 2)            | 0.30 | 0.30 | 1.50 | 0.50 | 0.55 | 0.20  | 0.80  |
| Simple Spread     | Dynamic Extreme (Agent 1)             | 1.20 | 1.20 | 1.20 | 0.50 | 0.69 | 0.20  | 0.80  |
| Simple Spread     | Dynamic Extreme (Agent 2)             | 0.30 | 0.30 | 1.50 | 0.50 | 0.69 | 0.20  | 0.80  |
| **Auction**       | CPT Agents                            | 0.88 | 0.88 | 2.25 | 0.61 | 0.69 | N/A   | N/A   |
| Auction           | Non-CPT Agents                        | N/A  | N/A  | N/A  | N/A  | N/A  | N/A   | N/A   |

Below are details on how to set these parameters for each specific notebook:

---

**1. `CPT_MADDPG_Competitive.ipynb` (Simple Tag with Linearized CPT)**

* **Core CPT Approach:** Applies a linearized CPT model to all agents (chasers and evaders) in the "simple_tag" environment. The CPT modifications influence both actor and critic losses.
* **Setting Hyperparameters:**
    * **`α` (Alpha for gains), `β` (Alpha for losses):** Modify the `alpha` parameter within the `u_plus(x)` and `u_minus(x)` functions. The script uses a single `alpha` in `u_plus` and a different `alpha` in `u_minus`.
        ```python
        def u_plus(x):
            alpha = 0.88 # <-- Set table's α here
            return torch.pow(x, alpha)

        def u_minus(x):
            alpha = 0.88 # <-- Set table's β here
            lam = 2.25   # <-- Set table's λ here
            return lam * torch.pow(-x, alpha)
        ```
    * **`λ` (Lambda - Loss Aversion):** Modify the `lam` variable within the `u_minus(x)` function (see above).
    * **`(w⁺)'`, `(w⁻)'` (Linearized Weighting Sensitivities):** Modify the global constants `w_plus_prime_const` and `w_minus_prime_const`.
        ```python
        w_plus_prime_const = 0.20 # <-- Set table's (w⁺)' here
        w_minus_prime_const = 0.80 # <-- Set table's (w⁻)' here
        ```
    * The `γ` and `δ` parameters from the table (related to Prelec's `eta`) are implicitly used to derive these constant sensitivities. The script does not dynamically use `w_plus_prime(p)` or `w_minus_prime(p)` in the loss.
* **Note:** The `iteration_when_stop_training_evaders` variable can be set to control when the "agent" group (evaders) stops training.

---

**2. `CPT_MADDPG_Cooperative.ipynb` (Simple Spread with Agent-Specific Linearized CPT)**

* **Core CPT Approach:** Applies a linearized CPT model to a *specific designated agent* (e.g., `agent_0`) in the "simple_spread" cooperative environment. Other agents use standard DDPG. Agents are treated as individual groups.
* **Setting Hyperparameters for the CPT Agent:**
    * Identify the `cpt_agent` variable (e.g., `cpt_agent = "agent_0"`). The CPT loss is applied only to this agent.
    * **`α` (Alpha for gains), `β` (Alpha for losses):** Modify the `alpha` parameter within the `u_plus(x)` and `u_minus(x)` functions, similar to the competitive notebook.
        ```python
        def u_plus(x):
            alpha = 0.88 # <-- Set table's α for the CPT agent
            return torch.pow(x, alpha)

        def u_minus(x):
            alpha = 0.88 # <-- Set table's β for the CPT agent
            lam = 2.25   # <-- Set table's λ for the CPT agent
            return lam * torch.pow(-x, alpha)
        ```
    * **`λ` (Lambda - Loss Aversion):** Modify `lam` in `u_minus(x)`.
    * **`(w⁺)'`, `(w⁻)'` (Linearized Weighting Sensitivities):** Modify the global constants `w_plus_prime_const` and `w_minus_prime_const`. These will apply to the designated CPT agent.
* **Note:** The environment setup redefines `env.group_map` so each agent is its own group, enabling agent-specific loss functions.

---

**3. `CPT_MADDPG_QTE_Cooperative.ipynb` (Simple Spread with CPT Integral)**

* **Core CPT Approach:** Applies a CPT integral valuation (using a piecewise linear probability weighting function) to modify the actor's loss for *all* agents in the "simple_spread" cooperative environment. The critic learns standard Q-values.
* **Setting Hyperparameters:**
    * **`α`, `β`, `λ` (Utility Parameters):** The `u_plus(x)` and `u_minus(x)` functions define these globally. To make them impact the CPT integral in the actor loss, you would need to ensure the `final_returns` passed to `compute_cpt_integral` are first transformed by these utility functions. Currently, `compute_cpt_integral` in the actor loss seems to operate on raw returns.
        ```python
        def u_plus(x):
            alpha = 0.7 # <-- Set table's α
            return torch.pow(x, alpha)

        def u_minus(x):
            alpha = 0.95 # <-- Set table's β
            lam = 2.5    # <-- Set table's λ
            return lam * torch.pow(-x, alpha)
        ```
    * **Piecewise Linear Weighting Function (`L` parameters for `w_approx`):** This is the primary CPT component affecting the actor loss. The parameters `γ` and `δ` from your table would inform the shape of this piecewise function. You define this `L` list when initializing `CPTDDPGLoss`:
        ```python
        losses = {}
        for group, _agents in env.group_map.items():
            loss_module = CPTDDPGLoss(
                # ... other DDPG params
                w=(0, [1.9467, 0.0, 0.6967, 0.1113, 2.7097, -1.7097, 0.0890, 0.9046]) # <-- This list is L_params
            )
            # ...
            losses[group] = loss_module
        ```
        The `L` list structure is `[s1, i1, s2, i2, ..., sn, in, bp1, ..., bp(n-1)]`. The values for `γ` and `δ` (e.g., 0.61, 0.69) from the table for Prelec-like weighting would guide how you construct these slopes, intercepts, and breakpoints to approximate such a weighting function.
    * **`beta` (Scaling Factor in Actor Loss):** Modify the `beta` variable inside `CPTDDPGLoss.loss_actor`:
        ```python
        # Inside CPTDDPGLoss.loss_actor
        beta = 1.0 # <-- Tune this hyperparameter
        scale_factor = torch.exp(beta * phi_factor)
        loss_actor = - (scale_factor * Q_values).mean()
        ```

---

**4. `CPT_MADDPG_First_Price_Auction.ipynb` (First-Price Auction with CPT Integral)**

* **Core CPT Approach:** Simulates a first-price auction where one group of agents ("cpt_agents") uses the CPT integral with a piecewise linear probability weighting function to modify their actor loss. Other agents ("regular_agent") use standard DDPG.
* **Setting Hyperparameters for "cpt_agents":**
    * **`α`, `β`, `λ` (Utility Parameters):** Similar to the `CPT_MADDPG_QTE_Cooperative` notebook, these are defined in global `u_plus` and `u_minus` functions. For them to affect the CPT integral in the actor loss, `final_returns` would need pre-processing.
    * **Piecewise Linear Weighting Function (`L` parameters):** Set the `L` list in the `w` argument when `CPTDDPGLoss` is initialized for the "cpt_agents" group. The table shows N/A for `(w⁺)'` and `(w⁻)'` for auctions, which is correct as this notebook uses the integral approach for weighting. The `γ` and `δ` values from the table (0.61, 0.69 for CPT agents) would inform the construction of this `L` list.
        ```python
        # Inside the loop creating losses
        if group == "cpt_agents":
            loss_module = CPTDDPGLoss(
                # ...
                w=(0, [1.9467, 0.0, 0.6967, 0.1113, ...]) # <-- Set L_params based on γ, δ
            )
        ```
    * **`beta` (Scaling Factor in Actor Loss):** Modify `beta` within `CPTDDPGLoss.loss_actor`.
* **Note:** The "regular_agent" group will use the standard `DDPGLoss`.

---

**5. `MADDPG_Cooperative_Adaptive_Behavioral_Parameters.ipynb` (Simple Spread with Adaptive CPT)**

* **Core CPT Approach:** Agents in "simple_spread" have *learnable* CPT parameters (`alpha`, `lam`, and `gamma` parameters for dynamic probability weighting functions). These are updated via a secondary loss.
* **Setting Initial Hyperparameters:** The table values for "Dynamic" variants correspond to the *initial values* for these learnable parameters.
    * **`init_alpha`, `init_lam`, `init_w_plus_gamma`, `init_w_minus_gamma`:** Set these when creating `AdaptiveBehavioralParameters` instances for each agent in the `adaptive_params` ModuleDict.
        ```python
        adaptive_params = nn.ModuleDict({
            "agent_0": AdaptiveBehavioralParameters(
                init_alpha=0.70, # <-- Table's α for Dynamic (Agent 1)
                init_lam=2.50,   # <-- Table's λ for Dynamic (Agent 1)
                init_w_plus_gamma=0.61, # <-- Table's γ for Dynamic (Agent 1)
                init_w_minus_gamma=0.69 # <-- Table's δ for Dynamic (Agent 1)
            ),
            "agent_1": AdaptiveBehavioralParameters(
                init_alpha=0.65, # <-- Table's α for Dynamic (Agent 2)
                # ... and so on for lam, w_plus_gamma, w_minus_gamma
            ),
        })
        ```
    * The table columns `(w⁺)'` and `(w⁻)'` are used as initial constants if the dynamic weighting functions were to fall back or be compared against a linearized version, but the primary mechanism here is learning `w_plus_prime_gamma` and `w_minus_prime_gamma`.
* **Controlling Adaptation:**
    * `optimizer_behavioral` learning rate.
    * `adaptive_update_frequency`, `freeze_adaptive_until`.
    * `scale_factor` and `reg_lambda` for the adaptive loss.

---

**6. `MADDPG_Cooperative_Allowing_Observe_Utility_Adjusted_Rewards.ipynb` (Simple Spread with Observability of CPT Utilities)**

* **Core CPT Approach:** Agents in "simple_spread" have fixed, agent-specific CPT parameters. The key is that the CPT transformations (`C_transform_cross`, `compute_phi_cross`) average CPT-adjusted values *across agents*, implying agents' learning is influenced by the collective (CPT-adjusted) utility.
* **Setting Hyperparameters:**
    * These are set in the `agent_params` dictionary, where each agent gets its own set of CPT values.
        ```python
        agent_params = {
            "agent_0": { # Corresponds to "Observability CPT (Seeing – RS Agent)" or similar
                "alpha": 0.70,  # <-- Table's α
                "lam": 0.80,    # <-- Table's λ
                "w_plus_prime_const": 0.80,  # <-- Table's (w⁺)'
                "w_minus_prime_const": 0.20, # <-- Table's (w⁻)'
            },
            "agent_1": { # Corresponds to "Observability CPT (Seeing – RA Agent)" or similar
                "alpha": 0.65,
                "lam": 2.80,
                "w_plus_prime_const": 0.25,
                "w_minus_prime_const": 0.75,
            },
        }
        ```
    * The `alpha` in `agent_params` is used for both gain and loss curvature in `u_plus_agent` and `u_minus_agent`. Your table has separate `α` and `β`; you'll need to decide if `agent_params["alpha"]` corresponds to the table's `α` or if you need to modify the utility functions to accept separate alpha_gain and alpha_loss.
    * The `γ` and `δ` parameters from the table are implicitly represented by the `w_plus_prime_const` and `w_minus_prime_const`.

By following these guidelines for each notebook, you should be able to replicate the experiments detailed in your report. Remember to ensure that the general RL hyperparameters (learning rates, batch sizes, etc.) are also set as per your experimental setup.



# Executable Demos Links

The following are the links to the various demos of the Notebooks in the final notebooks section:

- Naive Approximation to CPT Integral with Competitive Simple Tag Environment: [Click Here](https://colab.research.google.com/drive/1CKJj0NpJg01_s5ZBLtaq54yZXgXHgXHv?usp=sharing)

- Naive Approximation to CPT Integral with Cooperative Simple Spread Environment: [Click Here](https://colab.research.google.com/drive/19j_OX9h3ILSOxr5UyM3RoRQCBX6GU7mk?usp=sharing)

- Improved Approximation to CPT Integral with First Price Auction: [Click Here](https://colab.research.google.com/drive/1XgJex1OQm6uNNzXqUVjoxRJBeLvh3Rv5?usp=sharing)

- Improved Approximation to CPT Integral with Cooperative Simple Spread Environment: [Click Here](https://colab.research.google.com/drive/1LahMoXyqK_qw49q3vjJ9nlsvHSB4Oexk?usp=sharing)

- Naive Approximation with Dynamically Changing CPT Hyperparameters: [Click Here](https://colab.research.google.com/drive/1C-hR3uKY2O3y3gokATdnvAjj8NqQE6GY?usp=sharing)

- Naive Approximation with Agents Observing Utility Adjusted Rewards: [Click Here](https://colab.research.google.com/drive/1mLeeiy27eVut9JN038IwrBaL7954OuA7?usp=sharing)








