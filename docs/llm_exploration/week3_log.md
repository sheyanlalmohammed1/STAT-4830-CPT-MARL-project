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

- How does Cumulative Prospect Theory change equilibrium behavior in multi-agent optimization?

**Key Insights:**

## Techniques That Worked