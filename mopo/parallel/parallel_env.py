import gym
import mujoco_py as mjc
import pdb

from mopo.parallel.parallel_hopper import ParallelHopperEnv


if __name__ == '__main__':
	N = 10
	env = ParallelHopperEnv()
	
	# env = gym.make('Hopper-v2')
	# sim = env.unwrapped.sim
	# pool = mjc.MjSimPool.create_from_sim(sim, N)
	pdb.set_trace()