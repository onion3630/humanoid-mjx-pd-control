import os
import time
import math
import jax
from brax.training.agents.ppo import train as ppo
from brax.io import model
from pd_env import PDHumanoid

# ==========================================
# 🧠 学習ハイパーパラメータ
# ==========================================
TOTAL_TIMESTEPS = 15_000_000  # 総学習ステップ数（目標値）
LEARNING_RATE = 1e-4           
ENTROPY_COST = 1e-2            
NUM_ENVS = 2048                
BATCH_SIZE = 2048              
NUM_MINIBATCHES = 32           
SAVE_FILE = "humanoid_pd_params.pkg" 

# 評価の回数（ログ出力の回数）
NUM_EVALS = 11
# ==========================================

def main():
    # 共有ライブラリのパス設定
    if 'CONDA_PREFIX' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib:{os.getenv('LD_LIBRARY_PATH', '')}"

    # --- 終了ステップ数の切り上げ計算と警告 ---
    chunk_size = NUM_ENVS * 20 * NUM_MINIBATCHES  # 1回の最小計算ブロック (unroll_length=20)
    steps_per_eval = math.ceil(TOTAL_TIMESTEPS / NUM_EVALS / chunk_size) * chunk_size
    actual_steps = steps_per_eval * NUM_EVALS
    
    if actual_steps > TOTAL_TIMESTEPS:
        print(f"\n[注意] 指定ステップ数 ({TOTAL_TIMESTEPS:,}) はGPUの計算単位に満たないため自動調整されます。")
        print(f"       実際の最終ステップ数は約 {actual_steps:,} になる見込みです。\n")
    # ------------------------------------------

    print(f"使用デバイス: {jax.devices()}")
    env = PDHumanoid(backend='mjx')
    
    print(f"学習開始 ({actual_steps:,} steps予定)...")
    start_time = time.time()

    # --- 計算負荷ゼロのリッチな進捗表示 ---
    def progress(num_steps, metrics):
        reward = metrics.get('eval/episode_reward', 0.0)
        # 判明した正しい変数名で生存長を取得
        length = metrics.get('eval/avg_episode_length', 0.0) 
        elapsed = time.time() - start_time
        
        print(f"Step: {num_steps:>11,} | 報酬: {reward:>9.2f} | 生存長: {length:>6.1f} steps | 経過: {elapsed:>6.1f}s")
    # ----------------------------------------------

    # PPOによる学習の実行
    _, params, _ = ppo.train(
        environment=env,
        num_envs=NUM_ENVS,
        num_timesteps=TOTAL_TIMESTEPS,
        episode_length=1000,
        learning_rate=LEARNING_RATE,
        entropy_cost=ENTROPY_COST,
        batch_size=BATCH_SIZE,
        num_minibatches=NUM_MINIBATCHES,
        unroll_length=20,
        seed=0,
        progress_fn=progress,
        num_evals=NUM_EVALS
    )

    # 学習済みパラメータの保存
    model.save_params(SAVE_FILE, params)
    print(f"学習が終了しました。モデルを '{SAVE_FILE}' に保存しました。")

if __name__ == '__main__':
    main()