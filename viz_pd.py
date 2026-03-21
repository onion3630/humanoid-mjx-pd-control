import os
import jax
import mediapy as media
import mujoco
from mujoco import mjx
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

from pd_env import PDHumanoid

# ==========================================
# 🎬 視覚化パラメータ
# ==========================================
SIMULATION_STEPS = 1000        # 動画の長さ (約20秒)
LOAD_FILE = "humanoid_pd_params.pkg"     
OUTPUT_VIDEO = "humanoid_pd_walking.mp4" 
# ==========================================

def main():
    if 'CONDA_PREFIX' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib:{os.getenv('LD_LIBRARY_PATH', '')}"

    viz_env = PDHumanoid(backend='mjx')
    
    if not os.path.exists(LOAD_FILE):
        print(f"エラー: '{LOAD_FILE}' がありません。先に学習を実行してください。")
        return
        
    print(f"'{LOAD_FILE}' から学習済みパラメータを読み込んでいます...")
    params = model.load_params(LOAD_FILE)
    
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=viz_env.observation_size,
        action_size=viz_env.action_size,
        preprocess_observations_fn=running_statistics.normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    jit_inference_fn = jax.jit(make_inference_fn(params))
    
    jit_step_fn = jax.jit(viz_env.step)
    jit_reset_fn = jax.jit(viz_env.reset)

    key = jax.random.PRNGKey(0)
    state = jit_reset_fn(key)
    states = []

    print("推論を実行中...")
    for i in range(SIMULATION_STEPS):
        # 転倒時（done=1）は初期位置へリセットし、異常な状態の継続を防ぐ
        if state.done:
            state = jit_reset_fn(jax.random.PRNGKey(i))
            
        states.append(state)
        act, _ = jit_inference_fn(state.obs, jax.random.PRNGKey(i))
        state = jit_step_fn(state, act)

    # --- 【修正】MuJoCo公式レンダラーによる確実な描画 ---
    print("動画フレームをレンダリング中... (少し時間がかかります)")
    
    mj_model = viz_env.sys.mj_model
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    
    frames = []
    for s in states:
        # ✅ モデル(mj_model)とGPUデータ(s.pipeline_state)の2つを渡し、
        # CPUデータ(mj_data)を新しく受け取るのが正しい仕様です
        mj_data = mjx.get_data(mj_model, s.pipeline_state)
        
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

    print("MP4ファイルとして保存中...")
    # float() で囲むことで、JAX Arrayの型エラーを回避
    fps = int(1.0 / float(viz_env.dt))
    media.write_video(OUTPUT_VIDEO, frames, fps=fps)
    
    print(f"✨ '{OUTPUT_VIDEO}' を保存しました。メディアプレイヤーで再生してください。")

if __name__ == "__main__":
    main()