#%%
import numpy as np
from mazelab.generators import random_shape_maze
import gym

x = random_shape_maze(width=50, height=50, max_shapes=50, max_size=8, allow_overlap=False, shape=None)
# print(x)

empty = np.where(x == 0)
inds = np.random.choice(len(empty[0]), 2, replace=False)

start_idx = [[empty[0][inds[0]], empty[1][inds[0]]]]
goal_idx = [[empty[0][inds[1]], empty[1][inds[1]]]]

env_id = 'RandomShapeMaze-v0'


#%% Register env
from mazelab import MazeEnv

gym.envs.register(
  id=env_id,
  entry_point=MazeEnv,
  max_episode_steps=200,
  kwargs={"width":50, "height":50, "max_shapes":50, "max_size":8, "allow_overlap":False, "shape":None}
)

#%% show env
import matplotlib.pyplot as plt

env = gym.make(env_id)
env.reset()
img = env.render('rgb_array')
plt.imshow(img)


tasks = env.unwrapped.sample_tasks(3)

#%% solve
from mazelab.solvers import dijkstra_solver

impassable_array = env.unwrapped.maze.to_impassable()
motions = env.unwrapped.motions
start = env.unwrapped.maze.objects.agent.positions[0]
goal = env.unwrapped.maze.objects.goal.positions[0]
actions = dijkstra_solver(impassable_array, motions, start, goal)
print(actions)
env = gym.wrappers.Monitor(env, './', force=True)
rewards = 0.0
env.reset()
for action in actions:
    _, reward, _, _ = env.step(action)
    rewards += reward
env.close()
print(rewards)

#%% show
import imageio
from IPython.display import Image
from pathlib import Path
f = list(Path('./').glob('*.mp4'))[0]
reader = imageio.get_reader(f)
f = f'./{env_id}.gif'
with imageio.get_writer(f, fps=3) as writer:
    [writer.append_data(img) for img in reader]
Image(f)