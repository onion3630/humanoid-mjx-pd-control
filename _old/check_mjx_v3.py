import os
import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# --- WSL2上のライブラリパス自動修正 ---
if 'CONDA_PREFIX' in os.environ:
    conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    if conda_lib not in os.getenv('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{os.getenv('LD_LIBRARY_PATH', '')}"

def main():
    print(f"Checking JAX devices: {jax.devices()}")
    
    # 1. シンプルなモデル
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
        <option timestep="0.005" />
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
            <body name="upper_body" pos="0 0 1">
                <joint type="free"/>
                <geom type="capsule" size="0.1" fromto="0 0 0 0 0 0.4" rgba="0.5 0.5 0.8 1"/>
                <body name="lower_body" pos="0 0 0">
                    <geom type="capsule" size="0.08" fromto="0 0 0 0 0 -0.4" rgba="0.8 0.5 0.5 1"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """)
    
    mjx_model = mjx.put_model(model)
    
    # 並列数とステップ数
    num_envs = 2048
    steps = 500
    
    print(f"Parallel environments: {num_envs}")
    print(f"Steps per environment: {steps}")
    
    # データの初期化 (2048個分)
    data = mujoco.MjData(model)
    batch_data = jax.vmap(lambda _: mjx.put_data(model, data))(jnp.arange(num_envs))
    
    # --- 【重要】mjx.step を vmap して並列化する ---
    # これにより 1個用の step 関数が 2048個一括処理用に拡張されます
    parallel_step = jax.vmap(lambda d: mjx.step(mjx_model, d))

    @jax.jit
    def run_simulation(current_data):
        def scan_fn(carry, _):
            # 並列化された step を実行
            return parallel_step(carry), None
        
        # 500ステップ分、一気に時間を進める
        final_data, _ = jax.lax.scan(scan_fn, current_data, None, length=steps)
        return final_data

    # --- 実行 ---
    print("\n--- Compiling and Warming up... ---")
    start = time.time()
    batch_data = run_simulation(batch_data)
    # 計算が物理的に完了するまで待機
    jax.block_until_ready(batch_data)
    print(f"Compile + First Run time: {time.time() - start:.2f}s")

    print("\n--- Benchmark Run (The Real Power of RTX 4080) ---")
    start = time.time()
    batch_data = run_simulation(batch_data)
    jax.block_until_ready(batch_data)
    end = time.time()

    # --- 結果の表示 ---
    duration = end - start
    total_steps = num_envs * steps
    sps = total_steps / duration

    print(f"Benchmark Duration: {duration:.4f}s")
    print(f"Total Steps processed: {total_steps}")
    print(f"Steps Per Second (SPS): {sps:,.0f} FPS")
    print("-" * 40)

if __name__ == "__main__":
    main()