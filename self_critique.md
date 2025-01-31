# Self Critique

## Observe

After reading the report again for the first time, here are our initial takeaways:

- The report puts together the two big ideas that we had regarding everything we had read in Multi-Agent Reinforcement Learning and Cumulative Prospect theory and overall does a decent job of explaining the connection and interest we had betwen them.

- There is little to no takeaways in the content because of the lack of CPT integration into the multi-agent setting. We effectively haven't given any meaningful takeaways in the report other than it is possible to train a multi-agent reinforcement learning model and use an optimization aglorithm.


- The writing could use a bit of work. We need to more thoroughly define a lot of the things we are saying to make them understandable in a general context. We are talking heavily as though everyone reading the report has already studied the topic like we have.

## Orient

### Strengths

- We have proof of concept with Multi-Agent Reinforcement Learning optimization occuring within our code and we are able to effectively show that through the results in our code. 

- We have a really good understanding of the behavior that we hope to see in CPT-driven policy, especially when it comes to the final equilibrium point after training, so we can test effectively.



### Areas for Improvement

- Our mathematical formulation for the objective function that we are trying to optimize needs some work. Right now we are heavily relying on metrics used by authors of previous papers, but we need to come up with at least two metrics formally: a utility-based method and divergence based method. These will be the backbone of how we understand the effectiveness of our policy.

- We need to determine the tests (and environments) we would actually find extremely interesting to test agents under so that, if need be, we can modify the existing policy framework we are using to work on these environments .

- We basically have no defined stopping point to ensure our system has reached an optimal point, especially without the context of a specific winning metric. Without a defined stopping criterion or a strategy for stabilizing gradients, it’s hard to determine when the model has converged or reached optimal performance. 



### Critical Risks/Asssumptions

- We basically assume that CPT will be easily extendable to multi-agent settings, but the non-linear and non-stable structure of CPT in these environments will likely cause the need to handle a bunch of complexities associated  with destabalized training.

- We also assume that we will easily be able to implement the CPT algorithm using the codebase of the other authors. It is possible that the authors deny us access to this codebase in which case it would be on us to code the algorithm they developed from scratch and ensure it runs in the same way as indicated in their results.



## Decide

### Next actions

- Our first step of action will actually be implementing/coding the CPT-adjusted policy gradient theorem, enabling the design of a model-free policy gradient algorithm and allowing us to evaluate the training on an extremely simplified environment.

- We will write out the specific mathematical objectives we hope to minimize/maximize and will make sure they are in line with the policy gradient theorem

- We will introduce some criterion into our code to ensure that the model understands when training is complete and the agents effectively have learned the policy.

## Act


### Resource Needs

- We really need access to the codebase of the individuals who worked on the CPT policy gradient theorem in our report. The findings in that paper are key to what we aim to accomplish, so getting that as soon as possible is our highest priority.

- We need to work on getting an even better understanding of CPT in and of itself. We have an intermediate understanding of this topic, but understanding more about prospect theory would tell us how to measure its efffects even better in practice.

