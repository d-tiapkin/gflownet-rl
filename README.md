# Generative Flow Networks as Entropy-Regularized RL

Official code for the paper [Generative Flow Networks as Entropy-Regularized RL](https://arxiv.org/abs/2310.12934). 

Daniil Tiapkin*, Nikita Morozov*, Alexey Naumov, Dmitry Vetrov.

## Installation

- Create conda environment:

```sh
conda create -n gflownet-rl python=3.10
conda activate gflownet-rl
```

- Install PyTorch with CUDA:

```sh
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

## Hypergrids

Code for this part heavily utlizes library `torchgfn` (https://github.com/GFNOrg/torchgfn) and presented in the directory `hypergrid/`

Path to configurations (utlizes `ml-collections` library):

- General configuration: `hypergrid/experiments/config/general.py`
- Algorithm: `hypergrid/experiments/config/algo.py`
- Environment: `hypergrid/experiments/config/hypergrid.py`

List of available algorithms:
- Baselines: `db`, `tb`, `subtb` from `torchgfn` library;
- Soft RL algorithms: `soft_ql`, `munchausen_ql`, `sac`.

Example of running the experiment on environment with `height=20`, `ndim=4` with `standard` rewards, seed `3` on the algorithm `soft_dqn`.
```bash
    python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:soft_ql --env.height 20 --env.ndim 4
```
To activate learnable backward policy for this setting
```bash
    python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:soft_ql --env.height 20 --env.ndim 4 --algo.tied True --algo.uniform_pb False
```


## Molecules

Currently under construction 🚧 🔨

## Bit sequences

Examples of running baselines for word length `k=8`:

```
python3 bitseq/run.py --objective tb --k 8 --learning_rate 0.002
```

```
python3 bitseq/run.py --objective db --k 8 --learning_rate 0.002
```

```
python3 bitseq/run.py --objective subtb --k 8 --learning_rate 0.002 --subtb_lambda 1.9
```

Example of running `SoftDQN`:

```
python3 bitseq/run.py --objective softql --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 
```

Example of running `MunchausenDQN`:

```
python3 bitseq/run.py --objective softql --m_alpha 0.15 --k 8 --learning_rate 0.002 --leaf_coeff 2.0 
```

## Citation

```
@article{tiapkin2023generative,
  title={Generative Flow Networks as Entropy-Regularized RL},
  author={Tiapkin, Daniil and Morozov, Nikita and Naumov, Alexey and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:2310.12934},
  year={2023}
}
```
