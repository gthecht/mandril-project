# Mandril algorithm
## Brief summary of Mandril
### What is mandril
Mandril takes Maml, but instead of training on attempts, it trains on examples.

### How does it learn from examples
*Write how I use the expert to learn only if it got a reward.

## Learning from partially optimal experts
Since Mandril only creates a loss when the expert took a different action from the agent, and got a reward, I would expect the agent to learn only from the successful demonstrations of the expert. This would imply that when I have a partially optimal expert, I would expect the agent to replicate the better examples of the expert.

### The problem space - MAB
Describe the problem space - MAB with arms

## Experts
Describe the different types of experts.