# Week 3 LLM Exploration Summary


## Session Focus 

We used an LLM to explore some of the ideas related to how Cumulative Prospect Theory can be used to model agent decision-making in multi-agent environments, particularly in situations where agents do not behave as purely rational expected utility maximizers. We wanted to find out what was crucial in settings where risk sensitivity and probability distortions affect equilibrium behavior.

## Suprising Insights

### Conversation: Multi-Agent Optimization with CPT

**Prompt That Worked:** 

- Can CPT-based agents be more robust in adversarial MARL settings?

- How does Cumulative Prospect Theory change equilibrium behavior in multi-agent optimization?

**Key Insights:**

- The main thing that worked with these prompts was the fact that they were using language and rhetoric exemplified in the existing literature on this topic. The LLM seemed to understand exactly what we were asking in the context of the literature and give a very specific response (with evidence) rather than just simply saying some very basic response to the question we were asking.

- The LLM seemed to know a lot about aderversarial agents. It even suggested the idea of risk-distortion in the context of multi-agent settings, where adversarial policies in training would not be able to effectively be used to predict responses.


### Conversation: On-policy vs Off-policy RL

**Prompt That Worked:** 

- What are the practical trade-offs between on-policy and off-policy learning in MARL? Be specific and don't give me anything that would be a waste of my time to look into.

- "How does experience replay affect sample efficiency and policy stability in off-policy methods?

**Key Insights:**

- The LLM provided surprisingly nuanced insights into experience replay in off-policy RL. It highlighted that while replay buffers increase sample efficiency, they also introduce policy divergence issues, especially in non-stationary environments like multi-agent settings. This was particularly relevant because standard discussions of experience replay often assume a single-agent, stationary MDP, but the LLM correctly identified that in multi-agent RL, stale experience can misrepresent opponent strategies, leading to suboptimal policy updates.

- A particularly interesting insight was how exploration dynamics differ between on-policy and off-policy methods. The LLM pointed out that on-policy methods inherently preserve exploratory behavior, since they continually sample from the current policy. Off-policy methods, on the other hand, often require explicit exploration mechanisms (like entropy regularization or epsilon-greedy) to prevent them from prematurely exploiting suboptimal strategies. 

## Techniques That Worked

-  The main reason the prompts worked well was that they were phrased in a way that aligned with the terminology and framing used in reinforcement learning literature.

- We also made sure to tell the LLM not to give useless information or basic information which was helpful for more depth to the analysis.


## Dead Ends Worth Noting

- The biggest thing we noted when using the LLM was its lack of ability to mathematically explain things. It is very very good at explaining verbally but when we asked for the math to support what it said, it struggled to properly articulate the steps.

- The LLM seems to only know limited things about actual prospect theory and seems to have a very specific approach so forcing it to think deeper on the topic was not very fruitful.