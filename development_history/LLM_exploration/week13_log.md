# Week 13 LLM Exploration Summary


## Session Focus 

We used an LLM to help put together our final presentation and consolidate all of our results and connections for the class.

## Suprising Insights

### Conversation: Creating the Final Presentation (Literature Review, Results Summaries, Mathematical Objectives)

**Prompt That Worked:** 

- Draft a slide‑by‑slide outline for our final presentation covering: (1) motivation and literature review on CPT in multi‑agent RL, (2) our custom DDPG‑CPT loss derivation, (3) experimental setups (Simple Tag, Simple Spread, First‑Price Auction), (4) key quantitative results, and (5) conclusions and future directions.

- For each slide, generate 3–5 concise bullet points, linking our experimental plots back to the mathematical objectives (such as how risk aversion shaped evader tactics in Simple Tag).

**Key Insights:**

- The LLM’s outline forced us to articulate a clear narrative: starting from “why CPT?” and moving methodically through algorithm design, environments, and results—ensuring we highlighted how each experiment validated a theoretical claim.

- When asked for slide bullets, it deftly tied specific figures back to our core CPT parameters, making it easy to see which hyperparameter drove which behavioral shift.


## Techniques That Worked

- Framing the task as “draft slide outline” elicited structural guidance rather than free‑form prose, which saved us hours of reorganizing.

- Asking for bullets per slide—rather than full paragraphs—produced succinct talking points that fit nicely on PowerPoint frames.

## Dead Ends Worth Noting

- The LLM’s initial runs produced overly verbose speaker notes that didn’t match slide real estate.