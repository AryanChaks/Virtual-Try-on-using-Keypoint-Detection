import os
import subprocess

def call_virtual_try_on(viton_hd_path, virtual_env_path, run_name, input_image_path, output_image_path):
    env = os.environ.copy()
    env["PATH"] = virtual_env_path + "/bin:" + env["PATH"]

    os.makedirs(output_image_path, exist_ok=True)

    cmd = [
        os.path.join(virtual_env_path, 'bin/python'),
        os.path.join(viton_hd_path, "test_cpu.py"),
        '--name', run_name,
        '--dataset_dir', input_image_path,
        '--dataset_mode', 'inference',
        '--dataset_list', f'{run_name}.txt',
        '--checkpoint_dir', os.path.join(viton_hd_path, 'checkpoints'),
        '--save_dir', output_image_path
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running virtual try on: {e}")
        raise
    except Exception as e:
        print(f"Error processing virtual try on output: {e}")
        raise

