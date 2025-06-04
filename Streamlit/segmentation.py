import os
import subprocess

def get_segmentation_output(conda_path, segmentation_path, input_image_path, segmentation_image_dir):
    """Get segmentation output using shell command"""
    # Construct segmentation command

    env = os.environ.copy()
    env["PATH"] = conda_path + "/bin:" + env["PATH"]

    # Ensure the output directory exists
    os.makedirs(segmentation_image_dir, exist_ok=True)
    
    cmd = [
        os.path.join(conda_path, 'bin/python'),
        os.path.join(segmentation_path, "schp_utils/simple_extractor.py"),
        '--dataset', 'lip',
        '--model-restore', os.path.join(segmentation_path, 'schp_utils/checkpoints/exp-schp-201908261155-lip.pth'),
        '--input-dir', os.path.dirname(input_image_path),
        '--output-dir', segmentation_image_dir
    ]
    
    try:
        # Run segmentation
        subprocess.run(cmd, check=True, env=env)

    except subprocess.CalledProcessError as e:
        print(f"Error running segmentation: {e}")
        raise
    except Exception as e:
        print(f"Error processing segmentation output: {e}")
        raise
