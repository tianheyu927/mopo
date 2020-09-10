# Model-based Offline Policy Optimization (MOPO)

Code to reproduce the experiments in [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/pdf/2005.13239.pdf).



## Installation
1. Install [MuJoCo 2.0](https://www.roboti.us/index.html) at `~/.mujoco/mujoco200` and copy your license key to `~/.mujoco/mjkey.txt`
2. Create a conda environment and install mopo
```
cd mopo
conda env create -f environment/gpu-env.yml
conda activate mopo
# Install viskit
git clone https://github.com/vitchyr/viskit.git
pip install -e viskit
pip install -e .
```

## Usage
Configuration files can be found in `examples/config/`. For example, run the following command to run HalfCheetah-mixed benchmark in D4RL.

```
mopo run_local examples.development --config=examples.config.d4rl.halfcheetah_mixed --gpus=1 --trial-gpus=1
```

Currently only running locally is supported.


#### Logging

This codebase contains [viskit](https://github.com/vitchyr/viskit) as a submodule. You can view saved runs with:
```
viskit ~/ray_mopo --port 6008
```
assuming you used the default [`log_dir`](examples/config/halfcheetah/0.py#L7).

## Citing MOPO
If you use MOPO for academic research, please kindly cite our paper the using following BibTeX entry.

```
@article{yu2020mopo,
  title={MOPO: Model-based Offline Policy Optimization},
  author={Yu, Tianhe and Thomas, Garrett and Yu, Lantao and Ermon, Stefano and Zou, James and Levine, Sergey and Finn, Chelsea and Ma, Tengyu},
  journal={arXiv preprint arXiv:2005.13239},
  year={2020}
}
```
