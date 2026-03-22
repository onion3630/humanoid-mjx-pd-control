import os
import time
import datetime
import jax
import jax.numpy as jnp
from brax.training.agents.ppo import train as ppo
from brax.io import model
from pd_env import PDHumanoid, KP_GAIN, KD_GAIN, UPRIGHT_WEIGHT, HEIGHT_WEIGHT, STILL_THRESHOLD, STILL_SUCCESS_STEPS

# ==========================================
# ⚙️ 学習ハイパーパラメータ
# ==========================================
TOTAL_TIMESTEPS = 300_000_000 
NUM_ENVS = 256         
UNROLL_LENGTH = 128    
NUM_MINIBATCHES = 32           
LEARNING_RATE = 3e-4           
# ==========================================

def main():
    # タイムスタンプとベースファイル名の定義
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_NAME = "humanoid_pd_params"
    
    # 仕様変更: ベース名を用いてモデルとログのファイル名を紐付け
    SAVE_FILE = f"{BASE_NAME}_{ts}.pkg"
    LOG_FILE = f"{BASE_NAME}_trainlog_{ts}.csv"

    env = PDHumanoid(backend='mjx')
    action_repeat = getattr(env, 'action_repeat', 4)

    # 1回のログ出力で進むステップ数
    steps_per_update = NUM_ENVS * UNROLL_LENGTH * action_repeat
    num_evals = TOTAL_TIMESTEPS // steps_per_update
    
    # ログに表示し、最終的な正となるステップ数
    actual_steps = steps_per_update * num_evals
    
    # Brax内部のループ仕様に合わせるための逆算（4倍オーバーランの防止）
    brax_timesteps = actual_steps // action_repeat

    print(f"\n[構成確認]")
    print(f"       1チャンク(Update): {steps_per_update:,} steps")
    print(f"       総チャンク数 (表示回数): {num_evals}")
    print(f"       最終ステップ数: {actual_steps:,}")
    print(f"       出力モデル: {SAVE_FILE}")
    print(f"       出力ログ: {LOG_FILE}\n")

    start_time = time.time()
    last_step = -1

    # CSVヘッダーにメタデータを書き込み
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# MODEL_FILE: {SAVE_FILE}\n")
        f.write(f"# TOTAL_TIMESTEPS: {actual_steps}\n")
        f.write(f"# NUM_ENVS: {NUM_ENVS}\n")
        f.write(f"# LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"# KP_GAIN: {KP_GAIN}, KD_GAIN: {KD_GAIN}\n")
        f.write(f"# UPRIGHT_WEIGHT: {UPRIGHT_WEIGHT}, HEIGHT_WEIGHT: {HEIGHT_WEIGHT}\n")
        f.write(f"# STILL_THRESHOLD: {STILL_THRESHOLD}, STILL_SUCCESS_STEPS: {STILL_SUCCESS_STEPS}\n")
        f.write("Step,Reward,Length,Time\n")

    def progress(num_steps, metrics):
        nonlocal last_step
        
        if num_steps == last_step:
            return
        last_step = num_steps

        reward = metrics.get('eval/episode_reward', 0.0)
        length = metrics.get('eval/avg_episode_length', 0.0)
        elapsed_time = time.time() - start_time

        print(f"Step: {num_steps:<12,} | 報酬 (Reward): {reward:>10.2f} | 生存長 (Length): {length:>8.1f} | 経過 (Time): {elapsed_time:>8.1f}s")
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{num_steps},{reward:.2f},{length:.1f},{elapsed_time:.1f}\n")

    print("JITコンパイルを開始します... (これには2〜5分ほどかかります)")
    
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=brax_timesteps,  # 逆算した値を渡し、内部ループ回数を補正する
        episode_length=1000,
        num_envs=NUM_ENVS,
        unroll_length=UNROLL_LENGTH,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=4,
        learning_rate=LEARNING_RATE,
        entropy_cost=1e-2,
        discounting=0.97,
        gae_lambda=0.95,
        num_evals=num_evals,
        reward_scaling=0.1,
        progress_fn=progress
    )

    model.save_params(SAVE_FILE, params)
    print(f"\n学習完了。パラメータを {SAVE_FILE} に保存しました。")

if __name__ == "__main__":
    main()