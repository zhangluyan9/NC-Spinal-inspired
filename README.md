# Spinal-inspired artificial tactile interneuron with high-order burst spiking for intelligent edge interfaces

Code for the data-processing and SNN experiments in:

> Fanfan Li†, Zhanglu Yan†, J. Mao, G. Liu, H. Ren, B. Qin, Z. Zhang, H. Zhang, Y. Shen, Z. Zheng, W. Feng, D. Li, Y. Tang, S. Wang, Y. Jin, T. Luo, W.-f. Wong, Hong Wang\*, Bowen Zhu\*. *Spinal-inspired artificial tactile interneuron with high-order burst spiking for intelligent edge interfaces.* (†equal contribution; \*hongwang@xidian.edu.cn, zhubowen@westlake.edu.cn)

## Overview

A three-finger gripper records multimodal tactile signals (**force**, finger **position/angle**, **temperature**) while grasping 20 objects (2000 samples, 100/class, 50 timesteps each). The pipeline has two stages:

1. **`data_process/`** — map each tactile channel onto an *artificial tactile interneuron circuit* and simulate it to extract **burst-spiking features** (ISI / IBI / MBI per branch).
2. **`SNNmodel/`** — train a quantization-aware ANN on those features, fold BatchNorm, and convert it to a **rate-encoded SNN**; report ANN vs. SNN accuracy.

## Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

Plus **`catCuda`** — a custom CUDA spiking kernel (the only non-pip import, used by `SNNmodel/models.py`). It is not on PyPI: build it from the TNNLS-25 low-latency-SNN toolkit (`python setup.py install`; [needs an NVIDIA GPU, sm_60+, and a CUDA toolkit)](https://github.com/zhangluyan9/Low-latency-SNNs). 

> Stage 1 is pure NumPy/pandas and runs on CPU. **Stage 2 needs a GPU** (the SNN calls `.cuda()` via `catCuda`).

## Running

All scripts use **bare relative paths**, and the data + intermediate files live in the **experiment root one level up** . So either run scripts from there, e.g.

```bash
python data_process_step1.py
```

or copy the needed data files next to the scripts and run them in place.

### Stage 1 — tactile features (CPU)

```bash
python data_process_step1.py                          # raw CSV → normalized/projected (force,position)  → robot_data_projected.npz
python data_process_step1_resort.py                   # sort samples by class                            → robot_data_sorted.npz
python data_process_step2.py                          # build 9-D [force,position,temp]×3 vector          → robot_data_expanded_with_temp.npz
python data_process_step2_gothroughminefunction_low.py# run each triplet through the interneuron → ISI/IBI/MBI  → ..._simulated_function_low_...npz
python data_process_step3_training_testing.py         # NaN→0, reshape (2000,50,9)→(2000,450), 70/20/10 split → train/val/test_*.pt
```

`step2` and `gothroughminefunction_low` each read/write one filename set — edit the `np.load(...)` / `np.savez(...)` lines to pick a variant, and keep the **same suffix** through to the `.pt` files. (`step3_..._savecsv.py` exports `.csv` instead of `.pt`. Run `ISI_IBI_MBI_low_p.py` alone to print ISI/IBI/MBI for one sample triplet.)

**Modality variants** (the `step2_*` scripts hold some modalities at a constant to isolate one sense):

| Script | Force | Position | Temperature | Output suffix |
|---|:--:|:--:|:--:|---|
| `step2.py` | real | real | maps for classes 8 & 16† | `_with_temp` |
| `step2_force_angel.py` | real | real | const | `_force_angel` |
| `step2_tem_force.py` | real | const | const | `_force` |
| `step2_tem_angle.py` | const | real | const | `_angel` |
| `step2_tem_temper.py` | const | const | const‡ | `_temper` |

† `08.csv` overrides class **8**, `10.csv` overrides class **16** (the filename ≠ class label). ‡ As shipped, `step2_tem_temper.py` leaves temperature constant too (the map overrides are commented out) — un-comment them to actually drive the temperature branch. Constants and overrides are toggled by editing each script.

### Stage 2 — train ANN, convert to SNN, evaluate (GPU)

```bash
python SNN_classification_multi_runs_450.py --epochs 310 --T 31 --num-runs 5
```

Per run it trains a quantization-aware ANN (input 450 → `[384,512,768,128]` → 20; dropout `[0.10,0.10,0.35,0.25]`; AdamW lr 1e-3, wd 1e-4; cosine-warm-restart; label smoothing 0.1), folds BatchNorm into the linear weights, converts to the rate-encoded `Sparrow_SNN` (time window `T`, weight quantization index 8), and reports ANN vs. SNN test accuracy. Results (JSON + CSVs + figures) go to `results_450d_onlyforce/`.

Edit the three `torch.load(...)` lines in `single_run` to point at your variant's `.pt` files; change the output-directory name to match. Useful flags: `--epochs`, `--T` (SNN time window = activation quantization level), `--num-runs`. (`--lr` exists but `single_run` hard-codes AdamW lr 1e-3.)

The `*_rebuttal.py` (input 7500) and `*_rebuttal_nonspar.py` (input 147300) variants run the same pipeline on the **raw expanded signal** instead of the 450-D burst features.

## The interneuron model (`data_process/ISI_IBI_MBI_low_p.py`)

`simulate_neuro_circuit` models the interneuron as three parallel memristive branches (each `RL`, `Cm`, `Vin`) feeding a current-summing fusion node and an output spiking neuron, then extracts:

- **Mean ISI** — mean interval between consecutive spikes within a burst.
- **Mean IBI** — a within-burst sub-grouping interval (mean of `t[j+1]−t[j−1]` at local minima of the intra-burst ISI sequence), computed *inside* a burst, not between bursts.
- **Mean MBI** — mean interval between successive burst onsets (`diff(burst_starts)`).

`simulate_from_RLs(rl1, rl2, rl3)` fixes `Cm = [0.15, 5.5, 50] nF` and `Vin = 3 V`. Note the caller passes a **permuted** triplet — `simulate_from_RLs(temp, force, position)` — so temperature drives the 0.15 nF branch, force the 5.5 nF, position the 50 nF.

> `step2_gothroughminefunction.py` imports `ISI_IBI_MBI` (a higher-precision sibling not bundled here); the included `ISI_IBI_MBI_low_p.py` is the self-contained, runnable path.

## Model components (`SNNmodel/`)

- `units.py` — `Clamp_q_`: clamp to `[0,1]` then quantize to `q_level` (straight-through gradient).
- `models.py` — `Net` (training ANN: `Linear→BN→Clamp_q_→Dropout`), `InferenceNet` (BN-free), `Sparrow_SNN` (rate-encoded, `catCuda.getSpikes`), `SNN_LIF`.
- `merge_batchnorm.py` — folds each `Linear+BN` into one `Linear` for the SNN.

Seeds are fixed throughout (`step3`: 80/100; each SNN run: `3407+17+run_id`).
