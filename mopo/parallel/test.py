import numpy as np
import time
import gym
# import mujoco_py as mjc
import pdb

from mopo.parallel.parallel_hopper import ParallelHopperEnv

N = 10000
horizon = 1

env = gym.make('Hopper-v2')
parallel = ParallelHopperEnv(N)

env.reset()
t0_p = time.time()
obs = parallel.reset()
vec_qpos_init, vec_qvel_init = parallel.vec_qpos_qvel()
obs_p = [obs]
act_p = []
for i in range(horizon):
	act = [parallel.action_space.sample() for _ in range(parallel.N)]
	obs, rew, term, info = parallel.step(act)
	obs_p.append(obs)
	act_p.append(act)
obs_p = np.array(obs_p)
act_p = np.array(act_p)
t1_p = time.time()

##
t0_s = time.time()
obs_s = []
for sim in range(parallel.N):
	print(sim)
	qpos = vec_qpos_init[sim]
	qvel = vec_qvel_init[sim]
	actions = act_p[:,sim]

	env.set_state(qpos, qvel)
	obs = env.env._get_obs()
	obs_s.append([obs])
	for act in actions:
		obs, rew, term, info = env.step(act)
		obs_s[-1].append(obs)
t1_s = time.time()

## [ num_sims x horizon x obs_dim ] --> [ horizon x num_sims x obs_dim ]
obs_s = np.array(obs_s).transpose(1,0,2)
diff = np.abs(obs_p - obs_s).max()
print(obs_p.shape)

time_p = t1_p - t0_p
time_s = t1_s - t0_s

print('environments: {} | horizon: {} | diff: '.format(N, horizon, diff))
print('parallel: {} | sequential: {}'.format(time_p, time_s))
pdb.set_trace()
