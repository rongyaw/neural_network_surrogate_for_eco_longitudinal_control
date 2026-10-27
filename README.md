# SUMO Car-Following Traffic Simulation

This workspace contains SUMO-based traffic-following simulations with online
MPC, reconstructed explicit MPC, neural-network controllers, and IDM.

The main entry point is `scripts/run_sim.py`.

## Controller source files

Controller implementations are split by type under `scripts/controllers/`:

| File | Controllers |
| --- | --- |
| `baseline.py` | IDM baseline |
| `explicit_mpc.py` | Explicit MPC configuration, region model, and connected/unconnected controllers |
| `online_mpc.py` | CasADi car-following MPC |
| `neural.py` | Regular neural-network models and controller |
| `preview_nn.py` | Preview neural-network model and controller |

New code should import from `controllers`. `scripts/_controller.py` remains as
a compatibility facade for older scripts.

## Requirements

### System packages

- Python 3.8 or newer
- SUMO and SUMO tools
- A working TraCI installation

On Ubuntu, SUMO can be installed with:

```bash
sudo apt update
sudo apt install sumo sumo-tools
```

The current script expects SUMO at:

```text
/usr/bin/sumo
/usr/bin/sumo-gui
```

If SUMO is installed elsewhere, update the paths in `scripts/run_sim.py`.

### Python packages

Create an environment and install the packages used by the active simulation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib osqp traci torch casadi
```

For GPU use, install the PyTorch build matching the installed CUDA version
instead of the generic `torch` package if necessary.

The regular `NN` controller currently moves its model to CUDA, so it requires a
CUDA-enabled PyTorch installation and a visible GPU. The `PreviewNN` and
explicit-MPC controllers can use the CPU when CUDA is unavailable.

The repository also contains a ROS1 `package.xml`. ROS/catkin is only needed
for ROS package integration; it is not required for the direct
`scripts/run_sim.py` workflow.

## Setup

From the workspace root:

```bash
cd /home/tonyyaoyao/ANL_ws/src/sumo_cra_traffic_sim
source .venv/bin/activate
```

The active neural-network checkpoints are:

```text
scripts/traffic_following_control_dc_trained.pt
scripts/preview_traffic_following_control_conv_best.pt
```

## Running a simulation

General command format:

```bash
python3 scripts/run_sim.py [options] PROFILE CONTROLLER
```

Available driving-cycle profiles:

```text
Hwy, Nyc, Ftp, US06, FTPsec1, FTPsec2, FTPsec3
```

Available controllers:

| Controller | Description |
| --- | --- |
| `MPC` | CasADi online car-following MPC |
| `ExplicitUnconnected` | Reconstructed explicit MPC with locally estimated constant-acceleration PV preview |
| `ExplicitConnected` | Reconstructed explicit MPC with communicated/driving-cycle PV preview |
| `NN` | Regular neural-network controller using `traffic_following_control_dc_trained.pt` |
| `PreviewNN` | Preview neural-network controller using the local convolutional checkpoint |
| `IDM` | Intelligent Driver Model baseline |

Examples:

```bash
# CasADi online MPC
python3 scripts/run_sim.py --num_sv 5 --no_gui --print_level info Hwy MPC

# Explicit MPC with locally estimated PV preview
python3 scripts/run_sim.py --num_sv 5 --no_gui --print_level info Hwy ExplicitUnconnected

# Explicit MPC with connected PV preview
python3 scripts/run_sim.py --num_sv 5 --no_gui --print_level info Hwy ExplicitConnected

# Regular neural network
python3 scripts/run_sim.py --num_sv 5 --no_gui --print_level info Hwy NN

# Preview neural network
python3 scripts/run_sim.py --num_sv 5 --no_gui --print_level info Hwy PreviewNN

# IDM baseline
python3 scripts/run_sim.py --num_sv 5 --no_gui --print_level info Hwy IDM
```

Use `python3 scripts/run_sim.py --help` for the complete option list.

## Fleet initialization

For `Hwy`, `Nyc`, and `Ftp`, vehicles start at 1,200 m with 8 m
position spacing (not bumper-to-bumper clearance). Lane 0 is filled first:
`veh0` starts at 1,200 m and `veh150` at 0 m. If another vehicle would
have a negative position, `veh151` starts lane 1 at 1,197 m instead,
3 m behind `veh0`. Lane 1 continues at 8 m spacing through `veh300` at 5 m.
The two-lane capacity is 301 vehicles; larger requests are rejected.

Only `veh0` follows the selected driving cycle. The front vehicle in lane 1
(`veh151`) follows `veh0` using the selected controller, with a virtual +5 m
added to `veh0`'s position and position preview for this follower only (including
controller safety filtering). The physical starting separation is currently 3 m;
the controller initially sees 8 m. All other followers
use the preceding vehicle in their own lane, including for batched NN and IDM
control. Preview animation can select any follower, including `veh151`.
The state CSV and individual speed plot both track `veh1` following `veh0`.

CMI profiles retain their existing single-lane placement at `30 - 8 * i`;
the highway overflow and capacity checks do not apply to CMI.
All runs require at least two vehicles.

## CBF safety options

CBF safety filtering is enabled by default for explicit MPC and PreviewNN.
It can be disabled independently:

```bash
python3 scripts/run_sim.py --disable_explicit_cbf \
    --num_sv 5 --no_gui Hwy ExplicitConnected

python3 scripts/run_sim.py --disable_preview_cbf \
    --num_sv 5 --no_gui Hwy PreviewNN
```

The online CasADi MPC uses its optimization constraints and does not currently
apply the neural-network CBF filter.

## Explicit-MPC notes

The explicit controller is a reconstructed QP with its own implementation and
is not guaranteed to match the CasADi `MPC` controller exactly.

Current simulation settings are a 20-stage horizon, `dt=0.5`, and 10,000
active-set discovery samples. Regions are generated on the first run and
cached under:

```text
scripts/.explicit_mpc_cache/
```

Later runs load a matching cache instead of rebuilding the regions. The cache
key includes the controller mode and QP configuration. If the horizon,
weights, bounds, sample count, or other QP parameters change, a new cache is
created.

For details of the model, objective, constraints, region generation, OSQP
fallback, CBF filtering, and limitations, see
[Explicit MPC implementation details](Explicit_MPC_Implementation_Details.pdf).

## Runtime analysis

The runtime-analysis script runs several vehicle counts in sequence:

```bash
./runtime_analysis.sh CONTROLLER PROFILE MAX_VEHICLES VEHICLE_STEP PRINT_LEVEL CBF_MODE
```

For example:

```bash
./runtime_analysis.sh ExplicitConnected Hwy 100 10 quiet disabled
./runtime_analysis.sh MPC Hwy 100 10 quiet disabled
./runtime_analysis.sh NN Hwy 100 10 quiet disabled
./runtime_analysis.sh PreviewNN Hwy 100 10 quiet disabled
./runtime_analysis.sh IDM Hwy 100 10 quiet disabled
```

The runtime-analysis script currently supports:

```text
MPC, ExplicitConnected, NN, PreviewNN, IDM
```

The sixth argument controls explicit-MPC CBF behavior. Use `enabled` or
`disabled`. The script passes `--disable_explicit_cbf` only for
`ExplicitConnected` when `disabled` is selected.

When `--logging_sim` is enabled, runtime results are appended to:

```text
Runtime_<controller>.csv
```

Each row contains:

```text
vehicle_count, mean_ms, minimum_ms, maximum_ms, standard_deviation_ms
```

Runtime is measured over one complete SUMO/control step, including SUMO state
advancement and all follower-controller evaluations. Region generation occurs
before this timing interval. The deliberate pacing sleep is excluded.

The runtime is always reported in the console, but the runtime CSV is not
written unless `--logging_sim` is supplied. `runtime_analysis.sh` supplies this
flag automatically.

## Simulation logging and plots

To save per-step ego and preceding-vehicle states:

```bash
python3 scripts/run_sim.py --logging_sim \
    --num_sv 5 --no_gui Hwy ExplicitConnected
```

The simulation log is appended to:

```text
<profile>_<controller>.csv
```

Each row contains:

```text
simulation_time, ego_acceleration, ego_speed, ego_position,
preceding_acceleration, preceding_speed, preceding_position
```

To display the remaining speed plots:

```bash
python3 scripts/run_sim.py --plot_result \
    --num_sv 5 --no_gui Hwy ExplicitConnected
```

Traffic-density charts are no longer generated. The runtime CSV and state log
are independent of `--plot_result`.

## Headless and GUI operation

Use `--no_gui` for server or batch execution:

```bash
python3 scripts/run_sim.py --no_gui --num_sv 5 Hwy MPC
```

Without `--no_gui`, the script uses `sumo-gui` only when a display is
available.

## Useful files

```text
scripts/run_sim.py                  Main simulation entry point
scripts/controllers/                 Controller implementations by type
scripts/_controller.py               Compatibility exports for older imports
scripts/utils.py                    Driving-cycle and simulation utilities
runtime_analysis.sh                 Batch runtime analysis
speed_profile/                      Driving-cycle inputs
sumo/I-85_highway/                  Highway SUMO configuration
sumo/CMI/                           CMI SUMO configuration
```

## Troubleshooting

### Missing PreviewNN checkpoint

Confirm that this file exists:

```bash
ls -lh scripts/preview_traffic_following_control_conv_best.pt
```

### SUMO or TraCI import failure

Check both the executable and Python package:

```bash
which sumo
python3 -c "import traci; print(traci.__file__)"
```

### Explicit MPC starts slowly

The first run may generate thousands of active-set samples with OSQP. Wait
for region generation to finish. Subsequent runs should load the matching
file from `scripts/.explicit_mpc_cache/`.

### Regular NN CUDA error

The regular `NN` implementation currently uses CUDA tensors. Install a
CUDA-compatible PyTorch build and verify:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
