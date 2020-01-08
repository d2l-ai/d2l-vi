"""Integration with Sagemaker"""
import nbformat
from d2lbook.utils import run_cmd, find_files, split_config_str
from d2lbook.colab import insert_additional_installation, update_notebook_kernel

def generate_notebooks(config, eval_dir, sagemaker_dir):
    if not config['github_repo']:
        return
    # copy notebook fron eval_dir to colab_dir
    run_cmd(['rm -rf', sagemaker_dir])
    run_cmd(['cp -r', eval_dir, sagemaker_dir])
    notebooks = find_files('**/*.ipynb', sagemaker_dir)
    for fn in notebooks:
        with open(fn, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        update_notebook_kernel(notebook, config['kernel'])
        insert_additional_installation(notebook, config)
        with open(fn, 'w') as f:
            f.write(nbformat.writes(notebook))
