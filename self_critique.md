# Self Critique

## Observe

- **CPT integration** in MADDPG via a return wrapper applying subjective value and probability‐distortion steps before policy updates.
- **Experiments in two MPE settings**:
  1. **Competitive MPE (Simple Tag)**: Predator–prey task under varying CPT profiles.
  2. **Cooperative MPE (Simple Spread)**: Landmark coverage with risk‐sensitive coordination.
- **Auction extension**: First-price auction in both competitive and cooperative modes, showing CPT agents’ tendency to overbid.
- **Training stability** not fully achieved—extreme CPT configurations still diverge or oscillate.
- **Behavioral metrics** currently qualitative; lack precise convergence or loss-aversion indices.

## Orient

### Strengths

- **Human-like risk behavior**: CPT agents prefer sure payoffs in Simple Tag, adjust spacing in Simple Spread, and overbid in auctions.
- **Robust codebase**: Built on PyTorch/TorchRL/Vmas with modular CPT loss (`CPTDDPGLoss`), component functions (`u_plus`, `w_approx`), and CTDE critics.
- **Broad scenario coverage**: Included competitive MPE, cooperative MPE, dynamic CPT hyperparameter experiments, and auction simulations.

### Areas for Improvement

- **Learning stability**: Introduce gradient clipping or entropy regularization to tame non-convex CPT gradients.
- **Probability weighting**: Evaluate smooth approximations (e.g., parametric sigmoids) against the 6‐segment piecewise curve.
- **Hyperparameter search**: Automate joint tuning of RL (learning rate, τ) and CPT (α, λ, β) parameters.
- **Quantitative behavioral metrics**: Implement measures such as “loss‐aversion coefficient” and track training variance.

### Critical Risks/Assumptions

- Piecewise linear weighting may mis‐approximate true CPT curves, biasing agent behavior.
- MADDPG + CTDE stability fixes may not transfer to larger domains (e.g., LLM alignment).
- CPT’s non‐convex objectives could prevent scaling beyond toy tasks.

## Decide

### Next Actions

1. **Benchmark alternative weighting functions** (e.g., smooth power laws) against the piecewise approximation.  
2. **Incorporate stabilization techniques**: gradient clipping (ℓ₂‐norm < 1) and entropy regularization.  
3. **Automate hyperparameter search** (Optuna) for combined RL and CPT parameters, logging convergence stats.  
4. **Define and record behavioral metrics** per episode:
   - Sure-choice rate in coin-flip tasks.  
   - Spread distance variance in Simple Spread.  
   - Overbid ratio in auctions.

## Act

### Resource Needs

- **Reference implementations** of CPT-integrated MADDPG (Ewerhart & Leisen 2010).  
- **Extended GPU compute** for hyperparameter sweeps.  
- **Expert consultation** on realistic CPT parameter values.  
- **Recent literature** on stable nonconvex RB methods (e.g., trust-region variants).
