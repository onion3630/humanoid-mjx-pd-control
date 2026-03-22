import jax
import jax.numpy as jnp
from brax.envs.humanoid import Humanoid as BraxHumanoid
from brax.envs.base import State
from brax.base import math

# ==========================================
# ⚙️ Phase 1: 姿勢維持（ロバスト制御）パラメータ
# ==========================================
KP_GAIN = 60.0
KD_GAIN = 5.0
ACTION_LIMIT = 0.8

UPRIGHT_WEIGHT = 10.0
HEIGHT_WEIGHT = 5.0
HEALTHY_REWARD = 2.0
CTRL_COST_WEIGHT = 0.1

TARGET_HEIGHT = 1.2
FALL_HEIGHT = 0.7
FLY_HEIGHT = 2.5

PENALTY_REWARD = -100.0

# 🎯 静止ゴール用パラメータ
STILL_THRESHOLD = 0.05
STILL_SUCCESS_STEPS = 50.0
SUCCESS_BONUS = 500.0
# ==========================================

class PDHumanoid(BraxHumanoid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kp = KP_GAIN
        self.kd = KD_GAIN

    def reset(self, rng: jnp.ndarray) -> State:
        """初期状態にランダム性を加え、空中から落下させる (Domain Randomization)"""
        rng_pos, rng_qpos, rng_qvel = jax.random.split(rng, 3)
        
        qpos = self.sys.init_q
        qvel = jnp.zeros(self.sys.nv)

        # 1. 初期高度のランダム化
        z_offset = jax.random.uniform(rng_pos, (), minval=0.5, maxval=1.2)
        qpos = qpos.at[2].set(qpos[2] + z_offset)

        # 2. 初期姿勢・速度のランダム化
        joint_noise = jax.random.uniform(rng_qpos, (self.sys.nq - 7,), minval=-0.1, maxval=0.1)
        qpos = qpos.at[7:].add(joint_noise)
        
        qvel = qvel + jax.random.uniform(rng_qvel, (self.sys.nv,), minval=-0.05, maxval=0.05)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.action_size))

        # 静止カウントを0.0(float32)で初期化
        metrics = {'still_steps': jnp.zeros((), dtype=jnp.float32)}

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        target_angles = jnp.clip(action, -1.0, 1.0) * ACTION_LIMIT
        
        ps = state.pipeline_state
        
        curr_q = ps.qpos[7:] if hasattr(ps, 'qpos') else ps.q[7:]
        curr_v = ps.qvel[6:] if hasattr(ps, 'qvel') else (ps.qd[6:] if hasattr(ps, 'qd') else ps.v[6:])
        full_v = ps.qvel if hasattr(ps, 'qvel') else (ps.qd if hasattr(ps, 'qd') else ps.v)

        pd_control = self.kp * (target_angles - curr_q) - self.kd * curr_v
        safe_action = jnp.clip(pd_control, -1.0, 1.0)
        
        next_ps = self.pipeline_step(ps, safe_action)
        obs = self._get_obs(next_ps, action)

        torso_pos = next_ps.qpos[0:3] if hasattr(next_ps, 'qpos') else next_ps.q[0:3]
        torso_rot = next_ps.qpos[3:7] if hasattr(next_ps, 'qpos') else next_ps.q[3:7]

        # --- 静止判定ロジック ---
        v_norm = jnp.linalg.norm(full_v)
        is_still = v_norm < STILL_THRESHOLD
        
        current_still_steps = state.metrics.get('still_steps', jnp.zeros((), dtype=jnp.float32))
        next_still_steps = jnp.where(is_still, current_still_steps + 1.0, jnp.zeros((), dtype=jnp.float32))
        is_success = next_still_steps >= STILL_SUCCESS_STEPS
        bonus_reward = jnp.where(is_success, SUCCESS_BONUS, 0.0)

        # --- 既存報酬の計算 ---
        up_vec = math.rotate(jnp.array([0.0, 0.0, 1.0]), torso_rot)
        upright_reward = up_vec[2] * UPRIGHT_WEIGHT

        current_height = torso_pos[2]
        height_reward = jnp.exp(-jnp.square(current_height - TARGET_HEIGHT) * 2.0) * HEIGHT_WEIGHT

        is_fallen = jnp.logical_or(current_height < FALL_HEIGHT, current_height > FLY_HEIGHT)
        
        total_reward = upright_reward + height_reward + jnp.where(is_fallen, 0.0, HEALTHY_REWARD) - (jnp.sum(jnp.square(action)) * CTRL_COST_WEIGHT)
        
        # --- 報酬と終了判定の統合 ---
        reward = jnp.where(is_fallen, PENALTY_REWARD, total_reward + bonus_reward)
        is_done = jnp.logical_or(is_fallen, is_success)
        done = jnp.where(is_done, 1.0, 0.0)

        next_metrics = dict(state.metrics)
        next_metrics['still_steps'] = next_still_steps

        return state.replace(pipeline_state=next_ps, obs=obs, reward=reward, done=done, metrics=next_metrics)