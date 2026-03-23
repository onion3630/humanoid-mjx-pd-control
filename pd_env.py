# pd_env.py
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

# モータ仕様に基づく部位別の物理トルク制限
TARGET_MAX_TORQUE_TORSO = 1000.0  # 胴体・腹部 (3関節)
TARGET_MAX_TORQUE_LEG = 1000.0    # 股・膝 (左右計8関節)
TARGET_MAX_TORQUE_ARM = 1000.0    # 肩・肘 (左右計6関節)
REF_GEAR = 200.0

UPRIGHT_WEIGHT = 10.0
HEIGHT_WEIGHT = 5.0
HEALTHY_REWARD = 2.0

# ペナルティ重み（VRAM最適化・L1ノルム用）
P_ENERGY_WEIGHT = 0.005
P_VELOCITY_WEIGHT = 0.01
CROUCH_PENALTY_WEIGHT = 5.0

TARGET_HEIGHT = 1.2
FALL_HEIGHT = 0.7
FLY_HEIGHT = 2.5

PENALTY_REWARD = -100.0
# ==========================================

class PDHumanoid(BraxHumanoid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 実行時の除算オーバーヘッドを防ぐ事前計算
        # Brax Humanoidの17関節: 胴体(3), 右脚(4), 左脚(4), 右腕(3), 左腕(3)
        torques = jnp.array(
            [TARGET_MAX_TORQUE_TORSO] * 3 +
            [TARGET_MAX_TORQUE_LEG] * 8 +
            [TARGET_MAX_TORQUE_ARM] * 6
        )
        self.ctrl_limit = torques / REF_GEAR
        self.effective_kp = KP_GAIN / REF_GEAR
        self.effective_kd = KD_GAIN / REF_GEAR

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

        metrics = {}

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        target_angles = jnp.clip(action, -1.0, 1.0)
        
        ps = state.pipeline_state
        
        curr_q = ps.qpos[7:] if hasattr(ps, 'qpos') else ps.q[7:]
        curr_v = ps.qvel[6:] if hasattr(ps, 'qvel') else (ps.qd[6:] if hasattr(ps, 'qd') else ps.v[6:])
        full_v = ps.qvel if hasattr(ps, 'qvel') else (ps.qd if hasattr(ps, 'qd') else ps.v)

        # 実効ゲインを使用したPD制御計算（乗算のみ）
        pd_control = self.effective_kp * (target_angles - curr_q) - self.effective_kd * curr_v
        
        # 部位別のトルク上限配列（self.ctrl_limit）を用いた要素ごとのクリップ処理
        safe_action = jnp.clip(pd_control, -self.ctrl_limit, self.ctrl_limit)
        
        next_ps = self.pipeline_step(ps, safe_action)
        obs = self._get_obs(next_ps, action)

        torso_pos = next_ps.qpos[0:3] if hasattr(next_ps, 'qpos') else next_ps.q[0:3]
        torso_rot = next_ps.qpos[3:7] if hasattr(next_ps, 'qpos') else next_ps.q[3:7]

        # --- 終了判定ロジック (float32ベース) ---
        current_height_f32 = torso_pos[2]
        is_fallen = jnp.logical_or(current_height_f32 < FALL_HEIGHT, current_height_f32 > FLY_HEIGHT)

        # --- 報酬計算 (VRAM負荷低減のため bfloat16 で計算) ---
        bf_torso_pos = torso_pos.astype(jnp.bfloat16)
        bf_torso_rot = torso_rot.astype(jnp.bfloat16)
        bf_full_v = full_v.astype(jnp.bfloat16)
        bf_safe_action = safe_action.astype(jnp.bfloat16)

        up_vec = math.rotate(jnp.array([0.0, 0.0, 1.0], dtype=jnp.bfloat16), bf_torso_rot)
        upright_reward = up_vec[2] * jnp.bfloat16(UPRIGHT_WEIGHT)

        current_height = bf_torso_pos[2]
        k_height = jnp.bfloat16(2.0)
        height_error = current_height - jnp.bfloat16(TARGET_HEIGHT)
        
        # 指数関数(exp)を排除し、放物線近似を使用
        height_reward = jnp.maximum(jnp.bfloat16(0.01), jnp.bfloat16(1.0) - k_height * jnp.square(height_error)) * jnp.bfloat16(HEIGHT_WEIGHT)

        # 中腰ペナルティ計算 (一次式の差分による評価)
        crouch_diff = jnp.maximum(jnp.bfloat16(0.0), jnp.bfloat16(TARGET_HEIGHT) - current_height)
        p_crouch = crouch_diff * jnp.bfloat16(CROUCH_PENALTY_WEIGHT)

        # L1ノルムを使用したペナルティ計算
        p_energy = jnp.sum(jnp.abs(bf_safe_action)) * jnp.bfloat16(P_ENERGY_WEIGHT)
        p_velocity = jnp.sum(jnp.abs(bf_full_v)) * jnp.bfloat16(P_VELOCITY_WEIGHT)
        
        total_reward = upright_reward + height_reward + jnp.where(is_fallen, jnp.bfloat16(0.0), jnp.bfloat16(HEALTHY_REWARD)) - p_energy - p_velocity - p_crouch
        
        # --- 報酬と終了判定の統合 (float32へ復元) ---
        bf_reward = jnp.where(is_fallen, jnp.bfloat16(PENALTY_REWARD), total_reward)
        reward = bf_reward.astype(jnp.float32)

        done = jnp.where(is_fallen, 1.0, 0.0).astype(jnp.float32)

        next_metrics = dict(state.metrics)

        return state.replace(pipeline_state=next_ps, obs=obs, reward=reward, done=done, metrics=next_metrics)