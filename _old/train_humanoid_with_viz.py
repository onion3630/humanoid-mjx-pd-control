import os
import time
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.humanoid import Humanoid as BraxHumanoid
from brax.training.agents.ppo import train as ppo
from brax.io import html, model

# --- 1. PD制御を組み込んだカスタム環境の定義 ---
class PDHumanoid(BraxHumanoid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kp = 400.0  # 少し強めに設定
        self.kd = 40.0   

    def step(self, state, action):
        target_angles = action * 0.5 
        ps = state.pipeline_state
        
        # バックエンドの差（mjxのqpos vs springのq）を判定して取得
        if hasattr(ps, 'qpos'):
            curr_q, curr_v = ps.qpos[7:], ps.qvel[6:]
        else:
            curr_q, curr_v = ps.q[7:], ps.v[6:]

        pd_torque = self.kp * (target_angles - curr_q) - self.kd * curr_v
        return super().step(state, pd_torque)

# 環境の登録
if 'pd_humanoid' not in envs._envs:
    envs.register_environment('pd_humanoid', PDHumanoid)

def main():
    if 'CONDA_PREFIX' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib:{os.getenv('LD_LIBRARY_PATH', '')}"

    print(f"使用デバイス: {jax.devices()}")
    
    # --- 2. 学習フェーズ (MJX) ---
    env = envs.get_environment('pd_humanoid', backend='mjx')
    num_timesteps = 300_000_000 # 3億回
    
    print(f"PD制御ベース学習開始...")
    start_time = time.time()

    def progress(num_steps, metrics):
        reward = metrics['eval/episode_reward']
        print(f"Step: {num_steps:>11,} | 報酬: {reward:>8.2f}")

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_envs=2048,
        num_timesteps=num_timesteps,
        episode_length=1000,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        batch_size=2048,
        num_minibatches=32,
        unroll_length=20,
        seed=0,
        progress_fn=progress
    )

    model.save_params("humanoid_pd_params.pkg", params)
    print(f"学習完了（{time.time() - start_time:.1f}秒）")

    # --- 3. 視覚化フェーズ (Spring) ---
    print("\n🎬 視覚化実行中...")
    viz_env = envs.get_environment('pd_humanoid', backend='spring')
    
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_step_fn = jax.jit(viz_env.step)

    key = jax.random.PRNGKey(0)
    state = jax.jit(viz_env.reset)(key)
    states = []

    for i in range(1000):
        states.append(state)
        act, _ = jit_inference_fn(state.obs, jax.random.PRNGKey(i))
        state = jit_step_fn(state, act)

    output_filename = "humanoid_pd_walking.html"
    # レンダリングデータの構造を整理
    try:
        # 新形式への対応
        html.save(output_filename, viz_env.sys, [s.pipeline_state for s in states])
    except Exception:
        # 旧形式への対応
        html.save(output_filename, viz_env.sys, states)
    
    print(f"✨ '{output_filename}' 保存完了。")

if __name__ == "__main__":
    main()