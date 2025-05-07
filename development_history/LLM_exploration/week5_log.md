# Week 5 LLM Exploration Summary


## Session Focus 

We used an LLM to explore some ideas on how to get our multi-agent setting to work with a custom DDPG loss function centered towards CPT. We focused on both the practical insight into the algorithm as well as the code using the LLM.

## Suprising Insights

### Conversation: Understanding the DDPG Loss function in TorchRL

**Prompt That Worked:** 

- Explain to me in detail what the DDPG Loss function does in torchrl and all the components that go into it.

- What functionality would I have to fundementally change in this function to implement the algorithm that the authors of the MADDPG-CPT paper proposed? Be specific about the exact names of the components to change according to the documentation available for torchrl.

**Key Insights:**

- The prompts here specified the exact information we needed. We think by additionally focusing on the context we were interested in, which was torchrl, the LLM avoided getting confused into the generalities of the algorithm in other contexts.

- The LLM was actually really knowledgable on the topics, especially on torchrl, and we even asked it to look at the documentation for torchrl to confirm that it knew exactly what we were talking about. It helped the LLM clearly define the components to us and give us the clear insight on how to move forward.


### Conversation: Understanding and Coding the Utility and Linear Probability Weighting Functions

**Prompt That Worked:** 

- Give me the starter code to implement Kahneman and Traversky's Cumulative Prospect Theory utility and probability weighting functions. Make sure to specifiy the hyperparameters and explain the design choices you made in terms of the presentation of the integral.

- Explain the value of the hyperparameters found by the researchers, explain what each of them means and how to modify them to make an agent more or less risk averse.

**Key Insights:**

- One great insight was that the LLM had was that it was able to effectively give a really straightforward way to represent the exisitng utility functions and probability weighting functions as represented in the literature. It provided the code in a way that made it easy to see what was happening on the positive and the negative side of the rewards.

- Another insight was the LLM was able to actively explain how adjusting the hyperparameters of interest (such as lambda, alpha, etc.) would change what the utility and weighting functions represented. It also provided the optimal (human-like) values that were found by the researchers and explained what they meant in the context of a range of values.

## Techniques That Worked

-  The main reason the prompts worked well was that they were phrased in a way that got down to the root problem/question that we needed an answer to without too broadly asking for information. 

- We also made sure to tell the LLM not to give useless information or basic information which was helpful for more depth to the analysis.


## Dead Ends Worth Noting

- The biggest thing we noted when using the LLM was its lack of ability to dive too deep into torchrl. While it knew the fundamentals, it was particularly bad about giving answers to more depth of questions about each of the functionalities it spoke about.

- The LLM seems to only know limited things about actual prospect theory when it comes to mathematically, especially its components in cumulative prospect theory. For example, we found that it understood pretty well what the form of the utility and the probability weighting functions should be, but it could not provide us more depth when we asked how to put it in terms of a constant probability weighting.