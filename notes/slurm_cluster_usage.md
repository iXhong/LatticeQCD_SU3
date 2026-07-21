# Slurm Cluster Usage Notes

Date: 2026-07-19

## Cluster Access

Login node:

```bash
ssh ln01
```

Project directory on the cluster:

```text
/dssg/work/zhliu/projects/LatticeQCD_SU3
```

The initially suggested path
`/dssg/work/zhliu/projects/Lattice_SU3` was not present on `ln01`.

Scratch/output area used for server runs:

```text
/dssg/work/zhliu/scratch/lattice_su3/
```

Log directory:

```text
/dssg/work/zhliu/logs/
```

## Relevant Slurm Partitions

The pure-gauge update code is CPU/Numba based, so prefer CPU partitions.

Observed CPU partitions:

```text
cpunew          64 CPU cores per node
cpunew_sky      64 CPU cores per node
cpunew_sky_debug 64 CPU cores per node, 30 minute time limit
cpu_xnw         64 CPU cores per node
cpu_sky_big     96 CPU cores per node
```

For production work, use:

```text
cpunew_sky
```

For short tests, use:

```text
cpunew_sky_debug
```

## Required Runtime Environment

The project virtual environment uses Python 3.13.14 on the cluster:

```bash
.venv/bin/python --version
```

The GCC module is required before importing NumPy/Numba. Without it, NumPy can
fail with a missing `GLIBCXX_3.4.29` symbol from `libstdc++.so.6`.

Use this environment block in Slurm jobs:

```bash
module load gcc-11.2.0

export CC=gcc
export CXX=g++
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/uv-cache
```

Setting `NUMBA_NUM_THREADS` explicitly is important. In an interactive login
test after loading `gcc-11.2.0`, Numba reported 40 threads by default, which is
not appropriate inside a Slurm allocation unless 40 CPUs were requested.

## Server-Chain Workflow

Use `scripts/run_server_chain.py` for Slurm array jobs. Each array task runs one
independent Markov chain:

```bash
.venv/bin/python scripts/run_server_chain.py \
  CONFIG.toml \
  --chain "${SLURM_ARRAY_TASK_ID}" \
  --results-root RESULTS_ROOT \
  --resume
```

This writes scheduler-safe output under:

```text
RESULTS_ROOT/<run.name>/chains/chainNNN/
```

Each chain directory contains:

```text
manifest.json
observables.csv
configurations/*.npz
```

`--resume` is safe and should be kept in production scripts. Restarting a failed
or preempted task continues from the latest saved checkpoint in that chain.

## Parallelization Strategy

The best cluster-level scaling is independent chains first, then Numba threads
inside each chain.

Approximate CPU demand:

```text
active_array_tasks * SLURM_CPUS_PER_TASK
```

For 64-core nodes, a conservative production setting is:

```text
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39%16
```

This allows up to 16 chains concurrently, using about 64 CPU cores total when
fully packed on one node. Slurm may also spread tasks across nodes.

Avoid oversubscription such as many array tasks each allowing Numba to use all
visible cores.

## Smoke Test Record

Smoke configuration created on the cluster:

```text
configs/server_smoke_2x2x2x2.toml
```

Smoke Slurm script:

```text
slurm/server_smoke_2x2x2x2.sbatch
```

Test submission:

```bash
sbatch --partition=cpunew_sky_debug --time=00:10:00 \
  slurm/server_smoke_2x2x2x2.sbatch
```

Observed completed job:

```text
job id: 15793841
partition: cpunew_sky_debug
array: 0-1%2
cpus-per-task: 2
```

Both chains completed:

```text
chain000: completed_segment_sweeps=40, final_sweep=2040
chain001: completed_segment_sweeps=40, final_sweep=2040
```

Output was written under:

```text
/dssg/work/zhliu/scratch/lattice_su3/server_smoke_runs/server_smoke_2x2x2x2_b57/chains/
```

The stderr logs were empty.

## Production 10000-Configuration Plan

Production configuration created on the cluster:

```text
configs/server_prod_16x16x16x6_b57_10000.toml
```

Production Slurm script:

```text
slurm/server_prod_16x16x16x6_b57_10000.sbatch
```

Physics/run parameters:

```text
shape = [16, 16, 16, 6]
beta = 5.7
backend = "jit_checkerboard"
overrelaxation_sweeps = 2
chains = 40
sweeps_per_chain = 5200
discard_sweeps = 200
config_every = 20
```

Effective saved configurations:

```text
chains * ((sweeps_per_chain - discard_sweeps) / config_every)
= 40 * ((5200 - 200) / 20)
= 10000
```

Source thermalized configuration:

```text
results/runs/therm_16x16x16x6_b57_seed12345/configurations/chain00_hot_sweep001000.npz
```

The source file exists on the cluster and is about 6.8 MB. The production run
will likely write at least about 68 GB of NPZ configurations, plus manifests,
CSV files, and logs.

Submit production only after confirming queue policy and desired wall time:

```bash
cd /dssg/work/zhliu/projects/LatticeQCD_SU3
sbatch slurm/server_prod_16x16x16x6_b57_10000.sbatch
```

## Monitoring Commands

Queue:

```bash
squeue -u zhliu
squeue -j JOBID -o "%.18i %.18P %.30j %.8T %.10M %.6D %.4C %R"
```

Logs:

```bash
ls -lh /dssg/work/zhliu/logs/su3_prod_10k_JOBID_*.out
ls -lh /dssg/work/zhliu/logs/su3_prod_10k_JOBID_*.err
```

Output:

```bash
find /dssg/work/zhliu/scratch/lattice_su3/production_runs \
  -maxdepth 4 -type f -name "*.npz" | wc -l
```

Manifest status:

```bash
for f in /dssg/work/zhliu/scratch/lattice_su3/production_runs/*/chains/chain*/manifest.json; do
  echo "$f"
  .venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("status"), d.get("completed_segment_sweeps"), d.get("final_sweep"), d.get("chain"))' "$f"
done
```

## Safety Notes

- Do not run heavy production on the login node.
- Do not edit or remove existing results unless the run name and output root are
  confirmed.
- Keep production output in scratch or another large shared filesystem, not in
  the repository's tracked source tree.
- Keep `--resume` in server-chain jobs.
- Choose unique `run.name` values for new production runs. `run_server_chain.py`
  refuses incompatible resume manifests, but unique names make auditing simpler.
- The server-chain output layout is nested under `chains/chainNNN/`. Some older
  analysis scripts expect `results/runs/<run>/configurations/`; adapt or merge
  the output layout before analysis if needed.
- If a submitted array job is no longer wanted, cancel it explicitly:

```bash
scancel JOBID
```
