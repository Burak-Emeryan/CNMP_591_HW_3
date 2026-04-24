# CMPE 591 – HW4: Conditional Neural Movement Primitives (CNMP)

This repository implements a **Conditional Neural Movement Primitive** (CNMP) model for learning from demonstration, following the method from [Seker, Imre, Piater, Uğur (RSS 2019)](https://arxiv.org/abs/1906.10015).

## Environment Setup

Activate the conda environment (created for previous HWs):

```bash
conda activate cmpe591
```

Install dependencies (if not already installed):

```bash
pip install torch numpy matplotlib tqdm
pip install dm_control==1.0.10
pip install mujoco==2.3.2
pip install git+https://github.com/alper111/mujoco-python-viewer.git
```

## Data Collection

Collect trajectories from the MuJoCo simulation environment:

```bash
cd src
python collect_data.py
```

This saves trajectories to `data/trajectories.npz`.

## Training

Train the CNMP model:

```bash
python train.py --epochs 500 --lr 1e-4
```

After training, the script saves:
- `cnmp_model.pth` — trained model weights
- `figures/loss_curve.png` — training NLL loss vs epoch

## Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

This produces:
- `figures/mse_bar.png` — bar plot with End-Effector and Object MSE

## Project Structure

```
src/
├── homework4.py           # Course-provided environment & CNP reference
├── environment.py         # MuJoCo environment wrapper
├── mujoco_menagerie/      # Robot model assets
├── collect_data.py        # Data collection script
├── model.py               # CNMP model definition
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── utils.py               # Utility functions
figures/
├── loss_curve.png         # Training loss plot
└── mse_bar.png            # MSE evaluation bar plot
```

## Deliverables

1. `figures/loss_curve.png` — Training NLL loss vs epoch
2. `figures/mse_bar.png` — Bar plot with two bars (End-Effector MSE, Object MSE) with error bars
