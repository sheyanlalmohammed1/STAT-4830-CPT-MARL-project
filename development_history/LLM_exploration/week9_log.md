# Week 9 LLM Exploration Summary


## Session Focus 

We used an LLM to explore how we could implement some additional testing (specifically how we could test when two agents see each others utility functions and when they can choose their hyperparameters for their risk level dynamically).

## Suprising Insights

### Conversation: Learning about implemeting the test for seeing each others utility function.

**Prompt That Worked:** 

- Design an extension to the PettingZoo Simple Tag environment that allows each agent to be called at the start of each episode. How would you modify the environment API and agent interface to support this?

- Show me starter code for adding a callback in the agent’s step loop that stores the opponent’s utility parameters in the agent’s local state for use during action selection.

**Key Insights:**

- The LLM recommended injecting two new hooks into the environment class—before_episode() to broadcast each agent’s utility parameters, and on_step() in the agent wrapper to capture and cache them.


- It produced concrete sample code showing how to subclass the PettingZoo base environment, override step() to include a utilities field in the returned info dict, and update the agent’s observe() method to parse that field into its policy network’s input vector.


### Conversation: Learning about implementing the test for choosing hyperparameters dynamically.

**Prompt That Worked:** 

- In PettingZoo’s Simple Spread, write me a function select_hyperparams(history) that takes the last N episodes’ reward trajectories and returns updated CPT parameters. Show how to integrate this into the training loop.


- Explain how to schedule dynamic hyperparameter updates every M episodes in our RL training script, and what considerations—like smoothing or clipping—are needed to keep parameters in a valid range.

**Key Insights:**

- The LLM provided a clear template for a select_hyperparams() function: compute moving averages of returns, compare upside vs. downside variability, then adjust α and β by small learning‑rate–scaled deltas; apply a sigmoid transform to constrain them between (0,1).


## Techniques That Worked

- API‐design prompts: Asking “how would you modify the environment API” and “show me starter code” led the LLM to produce concrete code snippets instead of abstract descriptions.

- Integration‐focused follow‑ups: Framing questions around “integrate into the training loop” and “considerations like smoothing or clipping” drew out practical implementation advice and best practices.

## Dead Ends Worth Noting

- The LLM’s initial attempts to generate full end‑to‑end tests for these features were too monolithic—it conflated environment setup, model training, and evaluation in one script. We found it clearer to separate utility‑sharing from hyperparameter updates into two modular test suites.

- When asked to auto‑generate unit tests for these new callbacks, the LLM sometimes mis‑mocked the PettingZoo API, so we reverted to hand‑writing a few simple pytest functions to verify the hooks and clipping logic.