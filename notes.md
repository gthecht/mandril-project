# Notes
An explanation of what the code looks like in MAML

# Training
* baseline - *Linear baseline based on handcrafted features, as described in [1]*
* sampler - *Vectorized sampler to sample trajectories from multiple environements*
* metalearner - *Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning application, with an outer-loop optimization based on TRPO*

## MAML Algorithm
### The algorithm in the MAML paper

**Require:** $p(T)$: distribution over tasks
**Require:** $\alpha, \beta$: step size hyperparameters
* randomly initialize $\theta$
* **while** not done **do**
    * Sample batch of tasks $T_i ~ p(T)$
    * **for all T_i do**
    * Sample K trajectories $D=\{(x1, a1, ...x_H)\}$ using $f_\theta$ in $T_i$
        * Evaluate $\nabla_\theta L_{T_i}(f_\theta)$ using $D$ and $L_T$ in equation 4
        * Compute adapted parameters with gradient descent: $\theta' = \theta - \alpha \cdot \nabla_\theta L_{T_i}(f_\theta)$
        * Sample trajectories $D' = \{(x1, a1, ...x_H)\}$ using $f_{\theta'_i}$ in $T_i$
    * **end for**
    *	Update $\theta \leftarrow \theta - \beta \cdot \nabla_\theta\sum_{T_i\sim p(T)} L_{T_i}(f_{\theta_i'})$ using each $D'_i$ and $L_{T_i}$ in equation 4
*	**end while**

### The code as it is written
```python
for batch in trange(config['num-batches']):
  tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
  futures = sampler.sample_async(tasks,
                                  num_steps=config['num-steps'],
                                  fast_lr=config['fast-lr'],
                                  gamma=config['gamma'],
                                  gae_lambda=config['gae-lambda'],
                                  device=args.device)
  logs = metalearner.step(*futures,
                          max_kl=config['max-kl'],
                          cg_iters=config['cg-iters'],
                          cg_damping=config['cg-damping'],
                          ls_max_steps=config['ls-max-steps'],
                          ls_backtrack_ratio=config['ls-backtrack-ratio'])
```
As we can see, the sampler.sample is the part where we look at the training part - I'll need to change this in order to use the expert's example.
The metalearner.step (I presume where we do the step is the update to $\theta$)

---

## MandRIL algorithm
### The algorithm in the MandRIL paper
**Require:** $p(T)$: distribution over tasks
**Require:** $\alpha, \beta$: step size hyperparameters
* randomly initialize $\theta$
* **while** not done **do**
    * Sample batch of tasks $T_i ~ p(T)$
    * **for all T_i do**
    * Sample ~~K trajectories~~ *demos* $D^{tr}=\{(x1, a1, ...x_H)\}$ using $f_\theta$ in $T_i$
        * ~~Evaluate $\sout{\nabla_\theta L_{T_i}(f_\theta)}$ using $\sout{D}$ and $\sout{L_T}$ in equation 4~~
        * Instead:
            * Calculate the **inner loss**: $\frac{\partial L^{tr}_{T_i}}{\partial r_\theta}=$ MaxEntIRL-Grad$(r_\theta, T_i, D^{tr})$
            * Compute $\nabla_\theta L^{tr}_{T_i} (\theta)$ from $\frac{\partial L^{tr}_{T_i}}{\partial r_\theta}$
        * ~~Compute adapted parameters with gradient descent: $\sout{\theta' = \theta - \alpha \cdot \nabla_\theta L_{T_i}(f_\theta)}$~~
        * Instead:
            * Compute adapted **inner** parameters: $\phi_{T_i} = \theta - \alpha \cdot \nabla_\theta L_{T_i}(f_\theta)$ with gradient descent
        * ~~Sample trajectories $\sout{D' = \{(x1, a1, ...x_H)\}}$ using $\sout{f_{\theta'_i}}$ in $\sout{T_i}$~~
        * Instead:
            * Sample demos $D^{test}=\{\tau'_1,...,\tau'_{K'}\}\sim T_i$
            * Calculate the **outer loss**: $\frac{\partial L^{test}_{T_i}}{\partial r_\theta}=$ MaxEntIRL-Grad$(r_{\phi_{T_i}}, T_i, D^{test})$
            * Compute the meta-gradient: $\nabla_\theta L^{test}_{T_i}$ from $\frac{\partial L^{test}_{T_i}}{\partial r_\theta}$ via the chain rule
    * **end for**
    * ~~Update $\sout{\theta \leftarrow \theta - \beta \cdot \nabla_\theta\sum_{T_i\sim p(T)} L_{T_i}(f_{\theta_i'})}$ using each $\sout{D'_i}$ and $\sout{L_{T_i}}$ in equation 4~~
    * Instead, we take the gradients from within the for loop and:
        * Update $\theta \leftarrow \theta - \beta \cdot \sum_{T_i\sim p(T)} \nabla_\theta L^{test}_{T_i}(f_{\phi_{T_i}})$
*	**end while**

#### The function MaxEntIRL-Grad
* **Input:** Set of meta-trinaing tasks $\{T\}^{meta-train}$
* **Input:** hyperparameters: $\alpha, \beta$
* **Function** MaxEntIRL-Grad$(r_\theta,T,D)$
    * Compute state visitations of demos: $$\mu_D = \text{State-Visitations-Traj}(r_\theta, T)$$
    * Compute Max-Ent state visitations: $$E_T [\mu_T]=\text{State-Visitations-Policy}(r_\theta, T)$$ With the function from Ziebart.
    * $\frac{\partial L} {\partial r_\theta}=E_T [\mu_T]-\mu_D$
    * **Return** $\frac{\partial L} {\partial r_\theta}$
* **end function**
### The code
#### We have to change a few things:
1. Sample the demos instead of the trajectories:
    * We need to get the trajectories according to the demos instead of the reward function.
2. Get the inner loss using the max-entropy algorithm (for which we need to write the max-ent function)
3. Computing the gradient for the example training shouldn't be any different.
4. Sampling the test set, we need to change the following:
    * Once mroe use the max-ent function and from there to get the loss, and thence the gradient.

In addition we need to write the max-ent function.

#### Code to change:
1. In ***/policy.py:*** *def update_params*, change the parameters update from theta to phi.
2. In ***/multi_task_sampler.py:*** *def sample_trajectories*, we'll need to sample the demos from the expert instead of the trajectories of the model.
   Maybe here, instead of taking the actions from pi.sample, I can just take the demo's actions
3. In ***/multi_task_sampler.py:*** *def self.policy_lock*, we'll need to calculate the meta-training loss with the max-ent function, from the demos.
4. In ***/reinforcement_learning.py:*** *def reinforce_loss*, I'll need to change so that it uses max-ent (and implement that in a separate file)

#### 4.
the code:
```python
def reinforce_loss(policy, episodes, params=None):
    # I assume that pi the trajectory chosen by the policy. We need to change
    # this to the demos
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
                params=params)

    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)
    # I may want to return the dL/dr instead of L, since then we simply
    # differentiate by theta
    return losses.mean()
```
* **policy** is currently the CategoricalMLPPolicy, Which I assume has something to do with the MAB problem on which I'm trying it out.
* **episodes** are the maml_rl batch episodes, currently with 10 examples.
* **pi** is after running the policy, I cant see anything interesting there.
* **log_probs** looks like the log of the probabilities - which is exactly what we need.
What I should do is instead of calling this function, create a function for Mandril, that returns $\frac{\partial L_{T_i}}{\partial r_\theta}$

## TO DO
* I need to implement the max-ent: *Use $\frac{\partial L} {\partial r_\theta}=E_T [\mu_T]-\mu_D$ like in the max-ent function.* These are the state visitations trajecotries, and I can get them from Ziebart.
* Where to change the validation point for the meta testing.
* Build a framework for which I have demos, and can test mandril.
* Change how the loss is calculated (with max-ent of course)
* Change how the test step is done.