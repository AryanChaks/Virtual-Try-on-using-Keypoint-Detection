import os
import subprocess


def get_openpose_output(openpose_path, input_image_path, openpose_image_dir, openpose_json_dir):
    """Get body keypoints using OpenPose through shell command"""
    # Construct OpenPose command
    openpose_bin = os.path.join(openpose_path, 'build/examples/openpose/openpose.bin')

    os.makedirs(openpose_image_dir, exist_ok=True)
    os.makedirs(openpose_json_dir, exist_ok=True)
    
    # Set up command arguments
    cmd = [
        openpose_bin,
        '--image_dir', os.path.dirname(input_image_path),
        '--write_images', openpose_image_dir,
        '--write_json', openpose_json_dir,
        '--model_pose', 'BODY_25',
        '--net_resolution', '-1x368',
        '--render_threshold', '0.1',
        '--disable_blending', 'false',
        '--model_folder', os.path.join(openpose_path, 'models')
    ]
    
    try:
        # Run OpenPose
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running OpenPose: {e}")
        raise
    except Exception as e:
        print(f"Error processing OpenPose output: {e}")
        raise