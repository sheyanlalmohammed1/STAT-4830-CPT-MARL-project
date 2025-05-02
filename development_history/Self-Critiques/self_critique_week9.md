# Self Critique

## Observe

After reading the report again for the first time, here are our initial takeaways:

- The report/code finally is including the work we have done in the advanced MARL setting with CPT in multple enivironments while looking at an objective function which actually mathematically captures accurately what we want to measure and optimize. It also now include seeing other utility functions of other agents who are CPT trained as well as the case when the hyperparameters to the CPT are dynamic, adjusting iteratively to the actions of the cooperative agent.

- There are some takeaways in the content because we included details about CPT integration into the multi-agent setting. We can clearly now see a difference between the baseline integration with the integration of CPT into MARL setting.

- The writing could use a bit of work. We need to more thoroughly define a lot of the things we are saying to make them understandable in a general context. We are talking heavily as though everyone reading the report has already studied the topic like we have.

## Orient

### Strengths

- We have proof of concept with Multi-Agent Reinforcement Learning optimization occuring within our code and we are able to effectively show that through the results in our code. We also can show that our results are relevant to showing human-like behavior. 

- We now have a working idea of the experiments we need to run to show that CPT-optimized actions have occured and compare them to the results without the implementation of CPT. We have defined the cases we want to look at in the competitive and cooperative scenarios.

- We have a mathematical formulation that is actually being computed and working with predefined utility functions which we know have been tested and proved by researchers. 


### Areas for Improvement

- Our mathematical formulation for the objective function is being approximated in a pretty crude way. We need to try a couple of different approximations and then apply them to see the results and the way that the development changes of the rewards and actions that the agents experience in the competitive and cooperative settings.

- We can see that our system is coming to a nonstable equilibrium but the current MPE design has its limitations in interperable it is in terms of strategy. For example, how do we know one situation is overall better in simple spread than another except for average distance from the landmark. Clearly, this can create a case of manipulation where the lowest average distance with the highest reward isn't necessarily the best strategic outcome and creates issues in interpretaton.



### Critical Risks/Asssumptions

- We basically assume that CPT will be easily extendable to multi-agent settings, but the non-linear and non-stable structure of CPT in these environments will likely cause the need to handle a bunch of complexities associated  with destabalized training.



## Decide

### Next actions

- We need to continue the action of actually implementing/coding the CPT-adjusted policy gradient theorem, enabling the design of a model-free policy gradient algorithm and allowing us to evaluate the training on an extremely simplified environment.

- We will write out the specific mathematical objectives we hope to minimize/maximize and will make sure they are in line with the policy gradient theorem

- We will introduce some criterion into our code to ensure that the model understands when training is complete and the agents effectively have learned the policy.

## Act


### Resource Needs

- We really need access to the codebase of the individuals who worked on the CPT policy gradient theorem in our report. The findings in that paper are key to what we aim to accomplish, so getting that as soon as possible is our highest priority. It would be great to see how they implemented some of the approximations they discussed.

- We need to work on getting an even better understanding of CPT in and of itself. We have an intermediate understanding of this topic, but understanding more about prospect theory would tell us how to measure its efffects even better in practice.