from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from moplayground.envs.generic.mobase import MultiObjectiveBase
from moplayground.envs.dmcontrol.interface import CheetahInterface

class MOCheetah(MultiObjectiveBase):
    """Multi-Objective Cheetah Environment. 
    Objectives are speed, energy, and jumping height."""

    def __init__(
        self,
        env_params        : config_dict.ConfigDict,
        backend           : str,
        xml_path          : str = CheetahInterface.XML,
    ):
        super().__init__(
            xml_path          = xml_path,
            env_params        = env_params,
            backend           = backend,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_key, qvel_key = self._split(rng, 3)
        qpos = self._np.hstack([
            self._np.array(CheetahInterface.DEFAULT_FF),
            self._np.array(CheetahInterface.DEFAULT_JT)
        ])
        qvel = self._np.zeros(self.mj_model.nv)
        ctrl = self._np.zeros(self.mj_model.nu)
        
        data = self._data_init_fn(
            qpos         = qpos,
            qvel         = qvel,
            ctrl         = ctrl,
            time         = 0.0,
            xfrc_applied = self._np.zeros((self._mj_model.nbody, 6)),
        )
        parent_state = super().reset(
            rng            = rng,
            data           = data,
            history_length = self.params.history_length
        )
        info = {}
        info['xposbefore'] = 0.0
        info['xposafter']  = 0.01
        info['ang']        = 0.0
        info = parent_state.info | info

        done = self._np.array(0.0)
        rewards = self.reward_function(
            data   = data,
            action = ctrl,
            info   = info,
            done   = False,
        )
        reward, metrics = self.get_reward_and_metrics(rewards, {})
        
        obs = self._get_obs(data, parent_state.info)
        return self._state_init_fn(data, obs, reward, done, metrics, info)
    
    def state_vector(self, data):
        return self._np.hstack([data.qpos, data.qvel])

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info['xposbefore'] = state.data.qpos[0]
        action = self._np.clip(
            self.params.action_scale * action, 
            -1.0,
            1.0
        )
        data = self._step_fn(state.data, action)
        state.info['xposafter'] = data.qpos[0]
        state.info['ang']       = data.qpos[2]
        
        done = self.fall_termination(state.info)
        rewards = self.reward_function(
            data   = data,
            action = action,
            info   = state.info,
            done   = done
        )
        reward, metrics = self.get_reward_and_metrics(rewards, state.metrics)
        obs = self._get_obs(
            data,
            state.info
        )
        done = done.astype(float)
        return self._state_init_fn(data, obs, reward, done, metrics, state.info)
    
    def fall_termination(
        self,  
        info: dict
    ):
        return self._np.array(
            ~(abs(info['ang']) < self._np.deg2rad(50))
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        obs = self._np.concatenate([
            data.qpos[1:],
            self._np.clip(data.qvel, -10.0, 10.0),
        ])
        return {
            'state': obs,
            'privileged_state': obs
        }
    
    @property
    def action_size(self):
        return 6

    def reward_function(
        self,
        data,
        action,
        info,
        done
    ):
        rewards = {
            'alive'  : self.reward_alive(),
            'energy' : self.reward_energy(action),
            'height' : self.reward_height(data),
            'run'    : self.reward_run(info),
            'done'   : self.reward_done(done)
        }
        # Sparse variant of the run objective: emitted only when config opts in,
        # so dense configs keep an identical reward-key set (see get_metrics,
        # which requires a weight for every returned key).
        run_milestone_cfg = self.params.reward.get('run_milestone', None)
        if run_milestone_cfg is not None and run_milestone_cfg.get('enabled', False):
            rewards['run_milestone'] = self.reward_run_milestone(info)
        return rewards
    
    def reward_height(self, data: mjx.Data):
        return data.qpos[1] - CheetahInterface.DEFAULT_FF[1] + 0.2
    
    def reward_alive(self):
        return 1.0
    
    def reward_energy(self, action):
        # Per-step energy stays strictly positive when this constant exceeds the
        # max ||a||^2 (=6 for 6 clipped actuators), which removes the
        # fall-to-cut-cost exploit. Defaults to 4.0 for the dense baseline.
        energy_constant = self.params.reward.get('energy_constant', 4.0)
        return energy_constant - 1.0 * self._np.square(action).sum()
    
    def reward_run(self, info):
        reward_run = (info['xposafter'] - info['xposbefore']) / self.dt
        return self._np.min(self._np.array([4.0, reward_run]))

    def reward_run_milestone(self, info):
        # Sparse run reward: 0 every step until a new forward distance bucket is
        # crossed, then +1 per bucket (scaled by weights['run_milestone']).
        # Uses within-step positions so it auto-resets on episode boundaries.
        step_size     = self.params.reward.run_milestone.step_size
        before_bucket = self._np.floor(info['xposbefore'] / step_size)
        after_bucket  = self._np.floor(info['xposafter'] / step_size)
        return self._np.maximum(0.0, after_bucket - before_bucket)
    
    def reward_done(self, done):
        return self._np.array(done)
