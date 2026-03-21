import jax.numpy as jnp
from brax.envs.humanoid import Humanoid as BraxHumanoid

# ==========================================
# ⚙️ 環境・制御パラメータ（安定重視の調整）
# ==========================================
KP_GAIN = 60.0           # 筋力。姿勢の崩れを防ぎつつ支える強度
KD_GAIN = 5.0            # 制動。関節のガクつきを抑える
ACTION_LIMIT = 0.8       # 可動域。急激な姿勢変化を抑える

FALL_HEIGHT = 0.7        # 転倒判定。低すぎると判定される高さ
FLY_HEIGHT = 2.0         # 安全装置。異常な加速等で浮き上がったら即座に失敗とする

PENALTY_REWARD = -50.0   # 失敗時のペナルティ

# 報酬の重み（以前の宣言を維持）
FORWARD_WEIGHT = 2.0     # 前進報酬の係数
HEALTHY_REWARD = 2.0     # 生存ボーナス
CTRL_COST_WEIGHT = 0.01  # 制御コストの係数
# ==========================================

class PDHumanoid(BraxHumanoid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kp = KP_GAIN  
        self.kd = KD_GAIN  

    def step(self, state, action):
        # 1. アクションの正規化と制限
        target_angles = jnp.clip(action, -1.0, 1.0) * ACTION_LIMIT
        
        # 2. 現在の状態(ps)から PD制御入力を計算
        ps = state.pipeline_state
        if hasattr(ps, 'qpos'):
            curr_q, curr_v = ps.qpos[7:], ps.qvel[6:]
        else:
            curr_q = ps.q[7:]
            curr_v = ps.qd[6:] if hasattr(ps, 'qd') else ps.v[6:]

        pd_control = self.kp * (target_angles - curr_q) - self.kd * curr_v
        safe_action = jnp.clip(pd_control, -1.0, 1.0)
        
        # 3. 物理演算のみを実行して "next_ps" を取得
        next_ps = self.pipeline_step(state.pipeline_state, safe_action)
        
        # 4. 次の観測値(Observation)を生成
        obs = self._get_obs(next_ps, action)

        # 5. 次の状態から高さと速度を取得（報酬計算用）
        if hasattr(next_ps, 'qpos'):
            next_height = next_ps.qpos[2]
            x_velocity = next_ps.qvel[0] 
        else:
            next_height = next_ps.q[2]
            x_velocity = next_ps.qd[0] if hasattr(next_ps, 'qd') else next_ps.v[0]

        # 6. 転倒・異常な高さを検知
        is_fallen = jnp.logical_or(next_height < FALL_HEIGHT, next_height > FLY_HEIGHT)
        
        # 7. 独自報酬の計算
        # 速度報酬に上限(2.0m/s)を設ける
        clipped_velocity = jnp.clip(x_velocity, 0.0, 2.0)
        forward_reward = FORWARD_WEIGHT * clipped_velocity
        
        # 生存報酬と制御コスト
        healthy_reward = jnp.where(is_fallen, 0.0, HEALTHY_REWARD)
        ctrl_cost = CTRL_COST_WEIGHT * jnp.sum(jnp.square(action))
        
        # 合計報酬の算出
        reward = forward_reward + healthy_reward - ctrl_cost + jnp.where(is_fallen, PENALTY_REWARD, 0.0)
        
        # 8. 終了判定
        done = is_fallen.astype(jnp.float32)
        
        # 9. すべての情報をまとめた State オブジェクトを返す
        return state.replace(pipeline_state=next_ps, obs=obs, reward=reward, done=done)