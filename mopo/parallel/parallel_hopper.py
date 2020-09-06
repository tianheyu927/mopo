import os
from os import path
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.mujoco_env import convert_observation_to_space
import mujoco_py as mjc
import pdb

class ParallelMujocoEnv(mujoco_env.MujocoEnv):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, N):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(mujoco_env.__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mjc.load_model_from_path(fullpath)
        self.sim = mjc.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        #### parallel
        self.N = N
        self.pool = mjc.MjSimPool.create_from_sim(self.sim, self.N)
        ####

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done.any()

        self._set_observation_space(observation)

        self.seed()

    def _set_observation_space(self, observation):
        obs = observation[0]
        self.observation_space = convert_observation_to_space(obs)
        return self.observation_space

    def vec_do_simulation(self, vec_ctrl, n_frames):
        # pdb.set_trace()
        for sim, ctrl in zip(self.pool.sims, vec_ctrl):
            sim.data.ctrl[:] = ctrl
        # [sim.data.ctrl[:] = ctrl for sim, ctrl in zip(self.pool.sims, vec_ctrl)]
        # self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            # self.sim.step()
            self.pool.step()

    def set_state(self, qpos, qvel, sim_ind):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        sim = self.pool.sims[sim_ind]
        old_state = sim.get_state()
        new_state = mjc.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        sim.set_state(new_state)
        sim.forward()

    def vec_state_vector(self):
        qpos = self.query_pool(lambda x: x.data.qpos.flatten())
        qvel = self.query_pool(lambda x: x.data.qvel.flatten())
        return np.concatenate([qpos, qvel], axis=-1)

    def vec_qpos_qvel(self):
        vec_state = self.vec_state_vector()
        qpos_dim = self.sim.data.qpos.size
        vec_qpos = vec_state[:,:qpos_dim]
        vec_qvel = vec_state[:,qpos_dim:]
        return vec_qpos, vec_qvel
        # return np.concatenate([
        #     self.model.data.qpos.flat,
        #     self.model.data.qvel.flat
        # ])

class ParallelHopperEnv(ParallelMujocoEnv, utils.EzPickle):
    def __init__(self, N):
        ParallelMujocoEnv.__init__(self, 'hopper.xml', 4, N)
        utils.EzPickle.__init__(self)
        # self.N = 10
        # self.pool = mjc.MjSimPool.create_from_sim(self.sim, self.N)
        # pdb.set_trace()

    def query_pool(self, fn):
        val = [fn(sim) for sim in self.pool.sims]
        val = np.array(val)
        return val

    def step(self, a):
        if type(a) == list:
            a = np.array(a)
        if len(a.shape) == 1:
            a = np.array([a for _ in range(self.N)])

        vec_posbefore = self.query_pool(lambda x: x.data.qpos[0])
        self.vec_do_simulation(a, self.frame_skip)
        vec_qpos = self.query_pool(lambda x: x.data.qpos[0:3])
        vec_posafter, vec_height, vec_ang = vec_qpos[:,0], vec_qpos[:,1], vec_qpos[:,2]
        # np.array_split(vec_qpos, vec_qpos.shape[1], axis=-1)

        # posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        vec_reward = (vec_posafter - vec_posbefore) / self.dt
        vec_reward += alive_bonus
        vec_reward -= 1e-3 * np.square(a).sum(axis=-1)
        vec_s = self.vec_state_vector()
        vec_not_done = (np.isfinite(vec_s).all(axis=-1)) \
                   * (np.abs(vec_s[:,2:]) < 100).all(axis=-1) \
                   * (vec_height > .7) \
                   * (abs(vec_ang) < .2)
        vec_done = ~vec_not_done
        # done = not (np.isfinite(s).all(axis=-1) and (np.abs(s[2:]) < 100).all(axis=-1) and
        #             (vec_height > .7) and (abs(vec_ang) < .2))
        # pdb.set_trace()
        vec_ob = self._vec_get_obs()
        return vec_ob, vec_reward, vec_done, {}

    # def step(self, a):
    #     posbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     posafter, height, ang = self.sim.data.qpos[0:3]
    #     alive_bonus = 1.0
    #     reward = (posafter - posbefore) / self.dt
    #     reward += alive_bonus
    #     reward -= 1e-3 * np.square(a).sum()
    #     s = self.state_vector()
    #     done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
    #                 (height > .7) and (abs(ang) < .2))
    #     ob = self._get_obs()
    #     return ob, reward, done, {}

    def _vec_get_obs(self):
        fn = lambda x: np.concatenate([
            x.data.qpos.flat[1:],
            np.clip(x.data.qvel.flat, -10, 10)
        ])
        obs = self.query_pool(fn)
        return obs

        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     np.clip(self.sim.data.qvel.flat, -10, 10)
        # ])

    def reset_model(self):
        for n in range(self.N):
            self.reset_single(n)
        return self._vec_get_obs()

    def reset_single(self, sim_ind):
        # sim = self.pool.sims[sim_ind]
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel, sim_ind)
        # self.set_state(qpos, qvel)
        # return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



