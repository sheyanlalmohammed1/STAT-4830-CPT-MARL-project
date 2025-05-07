# Week 7 LLM Exploration Summary


## Session Focus 

We used an LLM to explore some ideas to begin testing our custom loss function on both the cooperative and competitive cases with the linear approximation to the integral.

## Suprising Insights

### Conversation: Understanding the Competitive Environment (Simple Tag) in the context of CPT

**Prompt That Worked:** 

- In Petting Zoo's Simple Tag environment, if we were to have one risk-averse evader go up against a risk-averse tagger, what would be expected to occur. 

- In Petting Zoo's Simple Tag environment, if we were to have one risk-seeking evador go up against a risk-seeking tagger, what would be expected to occur.

- Explain to me why we see the results we do in terms of the rewards, would this be consistent with the expectations under a CPT Policy.

**Key Insights:**

- Risk‐averse vs. risk‐averse: The LLM highlighted that both agents prioritize avoiding large negative payoffs (being caught), so the evader adopts highly conservative, predictable evasion routes, while the tagger focuses on minimizing potential “miss” scenarios. This leads to slow, cautious play and low‐variance but suboptimal mean rewards for both sides.

- Risk‐seeking vs. risk‐seeking: When both agents overweight tail events, the evader takes bold, unpredictable dashes toward corners, and the tagger makes aggressive intercept attempts. This produces high variance in episode returns—some runs end in rapid capture (very negative for the evader), others yield long chases (high positive for the evader).

- CPT‐consistent pattern: The LLM connected the observed reward distributions to CPT’s utility and weighting functions. It noted that under loss aversion (λ > 1), the risk‐averse evader overweights the risk of capture, compressing its strategy space, whereas the risk‐seeking pair overweights rare large gains (long evasion), leading to more exploratory behavior and bimodal payoff distributions.


### Conversation: Understanding the Cooperative Environment (Simple Spread) in the context of CPT

**Prompt That Worked:** 

- In Petting Zoo's Simple Spread environment, if we were to have two risk-averse agents working with one another, what would be expected to occur. 

- In Petting Zoo's Simple Spread environment, if we were to have two risk-seeking agents working with one another, what would be expected to occur.

- Explain to me why we see the results we do in terms of the rewards, would this be consistent with the expectations under a CPT Policy.

**Key Insights:**

- Risk‐averse cooperation: Both agents focus on avoiding the worst‐case scenario of leaving targets unattended. The LLM pointed out that they tend to cluster around safe zones, ensuring minimal coverage gaps but sometimes failing to cover all targets simultaneously. Reward variance is low but mean coverage is slightly below optimal.

- Risk‐seeking cooperation: Agents take on more divergent roles—one may sweep a cluster of targets aggressively while the other stays back as “backup.” This leads to occasional perfect coverage (high positive rewards) but also episodes with a few missed targets (steeper penalty).

- CPT perspective: The LLM explained that with probability weighting, risk‐averse agents overweight the chance of heavy penalties for missing any target, leading to conservative redundancy; risk‐seekers overweight the small chance of perfect coverage, driving split strategies and higher variance in team reward.

## Techniques That Worked

-  The main reason the prompts worked well was because the information that we were looking for was also not necessarily specific. It is hard to put an exact value on the result of what would happen under the CPT setting but generally understanding in the context of the game made the LLM able to explain more dpeth by the reward structure.

- In general, in giving the LLM the result plots, it was able to point out specific insights by the shape our reward resuls. This was extremely useful in the context of learning the results that matter and beign able to get some insight outside of just intuition of the results seen. 