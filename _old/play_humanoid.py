import os
import jax
from brax import envs
from brax.io import html, model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# --- WSL2上のライブラリパス自動修正 ---
if 'CONDA_PREFIX' in os.environ:
    conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    if conda_lib not in os.getenv('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{os.getenv('LD_LIBRARY_PATH', '')}"

def main():
    print(f"使用デバイス: {jax.devices()}")
    
    # 1. 環境の準備 (視覚化用)
    env_name = 'humanoid'
    viz_env = envs.get_environment(env_name, backend='spring')
    
    # 2. パラメータの読み込み
    model_path = "humanoid_pd_params.pkg" #"humanoid_params.pkg"
    if not os.path.exists(model_path):
        print(f"エラー: '{model_path}' が見つかりません。先に学習を行ってください。")
        return
        
    print(f"'{model_path}' から学習済みパラメータを読み込みます...")
    params = model.load_params(model_path)
    
    # 3. 脳（推論ネットワーク）の再構築
    # 学習時と同じ「PPO・観測正規化あり」の枠組みを作り、パラメータを流し込みます
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=viz_env.observation_size,
        action_size=viz_env.action_size,
        preprocess_observations_fn=running_statistics.normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_step_fn = jax.jit(viz_env.step)
    
    # 4. シミュレーションの実行 (学習ゼロ、推論のみ)
    print("ヒューマノイドを動かしています... (待ち時間は数秒です)")
    key = jax.random.PRNGKey(0)
    state = jax.jit(viz_env.reset)(key)
    states = [] 

    for i in range(1000):
        states.append(state)
        act, _ = jit_inference_fn(state.obs, jax.random.PRNGKey(i))
        state = jit_step_fn(state, act)

    # 5. HTML保存
    output_filename = "humanoid_playback.html"
    try:
        html.save(output_filename, viz_env.sys, states)
    except AttributeError:
        html.save(output_filename, viz_env.sys, [s.pipeline_state for s in states])
    
    print(f"✨ '{output_filename}' を保存しました。")

if __name__ == "__main__":
    main()