# Week 11 LLM Exploration Summary


## Session Focus 

We used an LLM to help create a custom Petting Zoo environment for a first price auction and then use the first price auction to simulate what occurs when CPT-trained agents go up against non-CPT trained agents (regular MADDPG).

## Suprising Insights

### Conversation: Learning about implemeting a custom environment in Petting Zoo and then using torchrl with it

**Prompt That Worked:** 

- Show me starter code for a PettingZoo environment named FirstPriceAuctionEnv where each of N agents submits a bid in [0,1], then pays their bid if they win the single item. Include both the reset() and step() methods, and specify the observation and action spaces

- Once FirstPriceAuctionEnv is defined, how would you wrap it in TorchRL’s RLModule API for training with MADDPG? Provide the skeleton of the Actor and Critic modules and the loss setup.

**Key Insights:**

- The LLM laid out a clear environment scaffold: using gym.spaces.Box([0],[1]) for bids, generating uniform valuations in reset(), computing utilities (valuation – bid if win, else zero), and returning an info dict with who won.

- It then demonstrated how to subclass TorchRL’s RLModule: defining separate Actor and Critic classes with simple MLPs, wiring the action distribution through a torch.distributions.Beta (to keep bids in (0,1)), and configuring MADDPG’s loss with the custom reward shape provided by the auction.


### Conversation: Learning how to seperate the training loop into groups of CPT-trained and non-CPT trained agents in torchrl

**Prompt That Worked:** 

- In a mixed‐policy scenario, how do I configure TorchRL’s MultiAgentCollector or training loop so that half the agents use CPT‑adjusted rewards and half use raw rewards? Show me pseudocode for assigning different RewardTransform modules per agent.

- Provide code for a training loop that alternates between updating CPT agents (with our custom CPT loss) and standard agents, ensuring their gradients don’t cross‐contaminate.

**Key Insights:**

- The LLM recommended using a dictionary of RewardTransform instances keyed by agent name, injecting a CumulativeProspectTransform for CPT agents and an identity transform for standard agents, then passing that dict into the MultiAgentCollector so each agent’s experience buffer carries the correct reward.

- For isolation during optimization, it suggested two separate optimizers and two LossModule instances, then in each training epoch running loss_cpt.backward() for CPT agents, stepping optim_cpt, zeroing its grads, and similarly for loss_standard and optim_standard. This clear separation prevents gradient intermingling and speeds convergence.

## Techniques That Worked

- Scaffold‐first prompts: Asking “show me starter code for…” helped the LLM provide minimal, runnable examples rather than verbose theoretical explanations.

- Modularization calls: Framing the question as “how do I configure X so that Y agents behave differently” elicited patterns for modular reward transforms and optimizer separation, which we could plug directly into our existing training script.

## Dead Ends Worth Noting

- The LLM’s first attempt conflated PPO’s single‐agent data pipeline with MADDPG’s multi‐agent collector, leading to incompatible API suggestions; we had to correct it by explicitly calling out MADDPG components.

- When we asked for a single unified training loop, the LLM generated code that tried to update both agent types in one pass. Splitting into two optimizer steps fixed the issue.