#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMT_ROOT="$ROOT_DIR/SMTgazer"
SIBYL_ROOT_DEFAULT="$ROOT_DIR/sibyl"
ENV_PREFIX="${SMTGAZER_ENV_PREFIX:-$ROOT_DIR/.venv}"
CACHE_DIR="$ROOT_DIR/.cache"
LOG_ROOT="$ROOT_DIR/logs"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$"

ACTION="check"
LIGHTGBM_BACKEND="auto"
TORCH_DEVICE="cuda:0"
DATASET=""
QUERY_ROOT=""
CHECKPOINT=""
SIBYL_ROOT="$SIBYL_ROOT_DEFAULT"
RUN_STAGE="full"
SEEDS_CSV="0"
CLUSTERS=""
OVERWRITE=0
TRAIN_SIBYL=0
GNN_EPOCHS=25
GNN_BATCH_SIZE=8
GNN_WORKERS=4
GRAPH_WORKERS=6
FEATURE_BATCH_SIZE=8
GNN_SEED=0
GPU_SAMPLE_INTERVAL=1

usage() {
    cat <<'EOF'
Usage:
  ./run_smtgazer.sh [check] [options]
  ./run_smtgazer.sh prepare [options]
  ./run_smtgazer.sh setup [options]
  ./run_smtgazer.sh run --dataset NAME [run options]

Actions:
  prepare     Install dependencies and build CUDA LightGBM without a GPU.
  setup       Create/check the Python environment and GPU LightGBM build.
  check       Run setup and the complete GPU smoke test (default).
  run         Run setup, GPU checks, and the real experiment immediately.

Common options:
  --lightgbm-backend auto|cuda|gpu  auto uses CUDA on WSL2 (default: auto).
  --torch-device DEVICE             Sibyl device (default: cuda:0).
  -h, --help                        Show this help.

Run options:
  --dataset NAME                    Dataset accepted by run_experiment.py.
  --stage full|features|portfolio   Experiment stage (default: full).
  --query-root PATH                 Override the default SMT2 query root.
  --checkpoint PATH                 Override the published model-0 checkpoint.
  --sibyl-root PATH                 Sibyl checkout (default: sibling ./sibyl).
  --train-sibyl                     Train a new Sibyl GNN on this GPU.
  --gnn-epochs N                    Sibyl training epochs (default: 25).
  --gnn-batch-size N                GPU graph training batch (default: 8).
  --gnn-workers N                   Training/inference data workers (default: 4).
  --graph-workers N                 Parallel SMT-to-graph workers (default: 6).
  --feature-batch-size N            Batched GPU embedding size (default: 8).
  --gnn-seed N                      Sibyl training seed (default: 0).
  --gpu-sample-interval SECONDS     GPU metric interval (default: 1).
  --seeds CSV                       Non-negative seeds, for example 0,1,2.
  --clusters N                      Override the original cluster upper bound.
  --overwrite                       Explicitly allow replacing canonical outputs.

Examples:
  ./run_smtgazer.sh prepare --lightgbm-backend cuda
  ./run_smtgazer.sh
  ./run_smtgazer.sh check --lightgbm-backend cuda
  ./run_smtgazer.sh run --dataset SyGuS --seeds 0
  ./run_smtgazer.sh run --dataset SyGuS --train-sibyl --seeds 0 --overwrite
  ./run_smtgazer.sh run --dataset SyGuS --stage portfolio --seeds 1,2,3
EOF
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "$value" && "$value" != --* ]] || die "$option requires a value"
}

if [[ $# -gt 0 && "$1" != -* ]]; then
    ACTION="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lightgbm-backend)
            require_value "$1" "${2:-}"
            LIGHTGBM_BACKEND="$2"
            shift 2
            ;;
        --torch-device)
            require_value "$1" "${2:-}"
            TORCH_DEVICE="$2"
            shift 2
            ;;
        --dataset)
            require_value "$1" "${2:-}"
            DATASET="$2"
            shift 2
            ;;
        --stage)
            require_value "$1" "${2:-}"
            RUN_STAGE="$2"
            shift 2
            ;;
        --query-root)
            require_value "$1" "${2:-}"
            QUERY_ROOT="$2"
            shift 2
            ;;
        --checkpoint)
            require_value "$1" "${2:-}"
            CHECKPOINT="$2"
            shift 2
            ;;
        --sibyl-root)
            require_value "$1" "${2:-}"
            SIBYL_ROOT="$2"
            shift 2
            ;;
        --train-sibyl)
            TRAIN_SIBYL=1
            shift
            ;;
        --gnn-epochs)
            require_value "$1" "${2:-}"
            GNN_EPOCHS="$2"
            shift 2
            ;;
        --gnn-batch-size)
            require_value "$1" "${2:-}"
            GNN_BATCH_SIZE="$2"
            shift 2
            ;;
        --gnn-workers)
            require_value "$1" "${2:-}"
            GNN_WORKERS="$2"
            shift 2
            ;;
        --graph-workers)
            require_value "$1" "${2:-}"
            GRAPH_WORKERS="$2"
            shift 2
            ;;
        --feature-batch-size)
            require_value "$1" "${2:-}"
            FEATURE_BATCH_SIZE="$2"
            shift 2
            ;;
        --gnn-seed)
            require_value "$1" "${2:-}"
            GNN_SEED="$2"
            shift 2
            ;;
        --gpu-sample-interval)
            require_value "$1" "${2:-}"
            GPU_SAMPLE_INTERVAL="$2"
            shift 2
            ;;
        --seeds)
            require_value "$1" "${2:-}"
            SEEDS_CSV="$2"
            shift 2
            ;;
        --clusters)
            require_value "$1" "${2:-}"
            CLUSTERS="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

case "$ACTION" in
    prepare|setup|check|run) ;;
    *) die "unknown action: $ACTION (expected prepare, setup, check, or run)" ;;
esac

case "$LIGHTGBM_BACKEND" in
    auto|cuda|gpu) ;;
    *) die "--lightgbm-backend must be auto, cuda, or gpu" ;;
esac

case "$RUN_STAGE" in
    full|features|portfolio) ;;
    *) die "--stage must be full, features, or portfolio" ;;
esac

validate_run_arguments() {
    [[ "$ACTION" == "run" ]] || return 0

    case "${DATASET,,}" in
        bmc) DATASET="BMC" ;;
        symex) DATASET="SymEx" ;;
        sygus) DATASET="SyGuS" ;;
    esac

    case "$DATASET" in
        BMC|Equality+LinearArith|QF_Bitvec|QF_LinearRealArith|QF_NonLinearIntArith|SyGuS|SymEx) ;;
        "") die "run requires --dataset" ;;
        *) die "unsupported dataset: $DATASET" ;;
    esac

    [[ "$SEEDS_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
        die "--seeds must be comma-separated non-negative integers"
    if [[ -n "$CLUSTERS" ]]; then
        [[ "$CLUSTERS" =~ ^[1-9][0-9]*$ ]] || \
            die "--clusters must be a positive integer"
    fi
    for positive in "$GNN_EPOCHS" "$GNN_BATCH_SIZE" "$GRAPH_WORKERS" "$FEATURE_BATCH_SIZE"; do
        [[ "$positive" =~ ^[1-9][0-9]*$ ]] || \
            die "GNN epochs/batches and graph workers must be positive integers"
    done
    [[ "$GNN_WORKERS" =~ ^[0-9]+$ ]] || die "--gnn-workers must be a non-negative integer"
    [[ "$GNN_SEED" =~ ^[0-9]+$ ]] || die "--gnn-seed must be a non-negative integer"
    [[ "$GPU_SAMPLE_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
        die "--gpu-sample-interval must be a positive number"
    [[ "$GPU_SAMPLE_INTERVAL" != "0" && "$GPU_SAMPLE_INTERVAL" != "0.0" ]] || \
        die "--gpu-sample-interval must be greater than zero"

    if (( TRAIN_SIBYL )) && [[ "$RUN_STAGE" == "portfolio" ]]; then
        die "--train-sibyl requires --stage full or features"
    fi
    if (( TRAIN_SIBYL )); then
        case "$DATASET" in
            BMC|SyGuS|SymEx) ;;
            *) die "GPU Sibyl training is available for BMC, SyGuS, and SymEx" ;;
        esac
        [[ -z "$CHECKPOINT" ]] || die "do not combine --train-sibyl with --checkpoint"
    fi

    if [[ "$RUN_STAGE" == "full" || "$RUN_STAGE" == "features" ]]; then
        case "$DATASET" in
            BMC|SymEx|SyGuS)
                QUERY_ROOT="${QUERY_ROOT:-$ROOT_DIR/sibyl/data/$DATASET}"
                if (( ! TRAIN_SIBYL )); then
                    CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/sibyl/inference/$DATASET/model_checkpoints/${DATASET}_model_0.pt}"
                fi
                ;;
        esac
        [[ -n "$QUERY_ROOT" ]] || die "$RUN_STAGE requires --query-root"
        QUERY_ROOT="$(readlink -m "$QUERY_ROOT")"
        [[ -d "$QUERY_ROOT" ]] || die "query root is not a directory: $QUERY_ROOT"
        if (( ! TRAIN_SIBYL )); then
            [[ -n "$CHECKPOINT" ]] || die "$RUN_STAGE requires --checkpoint or --train-sibyl"
            [[ -f "$CHECKPOINT" ]] || die "checkpoint is not a file: $CHECKPOINT"
            CHECKPOINT="$(readlink -f "$CHECKPOINT")"
        fi
        SIBYL_ROOT="$(readlink -m "$SIBYL_ROOT")"
        [[ -f "$SIBYL_ROOT/extract_embedding.py" ]] || \
            die "invalid Sibyl root: $SIBYL_ROOT"
    fi
}

validate_run_arguments

mkdir -p "$LOG_ROOT" "$CACHE_DIR/pip" "$CACHE_DIR/matplotlib"
LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"
export SMTGAZER_RUN_ID="$RUN_ID"
export SMTGAZER_LOG_DIR="$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

CURRENT_STAGE="startup"
on_error() {
    local line="$1"
    local command="$2"
    local status="$3"
    trap - ERR
    if (( BASH_SUBSHELL > 0 )); then
        exit "$status"
    fi
    printf '[ERROR] stage=%s line=%s exit=%s command=%q\n' \
        "$CURRENT_STAGE" "$line" "$status" "$command" >&2
    printf '[ERROR] Full log: %s\n' "$LOG_FILE" >&2
    exit "$status"
}
trap 'on_error "$LINENO" "$BASH_COMMAND" "$?"' ERR

log() {
    printf '[%s] %s\n' "$1" "$2"
}

log INFO "SMTgazer2 action=$ACTION run_id=$RUN_ID"
log INFO "Repository: $ROOT_DIR"
log INFO "Log: $LOG_FILE"
if [[ "$ACTION" == "run" ]]; then
    log INFO "Experiment: dataset=$DATASET stage=$RUN_STAGE seeds=$SEEDS_CSV"
    if [[ "$RUN_STAGE" == "full" || "$RUN_STAGE" == "features" ]]; then
        log INFO "Query root: $QUERY_ROOT"
        log INFO "Checkpoint: $CHECKPOINT"
    fi
fi

CURRENT_STAGE="repository validation"
for required in \
    "$SMT_ROOT/check_server.py" \
    "$SMT_ROOT/run_experiment.py" \
    "$SMT_ROOT/requirements-runtime.txt" \
    "$SMT_ROOT/smac/setup.py" \
    "$SIBYL_ROOT_DEFAULT/extract_embedding.py"; do
    [[ -f "$required" ]] || die "required repository file is missing: $required"
done

[[ "$(uname -s)" == "Linux" ]] || die "this script supports Linux only"
[[ "$(uname -m)" == "x86_64" ]] || die "this script supports x86_64 only"

AVAILABLE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR == 2 {print $4}')"
if [[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] && (( AVAILABLE_KB < 8 * 1024 * 1024 )); then
    die "at least 8 GiB free disk space is required for PyTorch and CUDA builds"
fi
log OK "Disk space check passed"

IS_WSL=0
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1
    log INFO "Detected WSL2"
fi

if [[ "$LIGHTGBM_BACKEND" == "auto" ]]; then
    if (( IS_WSL )); then
        LIGHTGBM_BACKEND="cuda"
    elif command -v clinfo >/dev/null 2>&1 && \
        clinfo 2>/dev/null | grep -Eiq 'Device Type.*GPU|CL_DEVICE_TYPE_GPU'; then
        LIGHTGBM_BACKEND="gpu"
    else
        LIGHTGBM_BACKEND="cuda"
    fi
fi
export SMTGAZER_LIGHTGBM_DEVICE_TYPE="$LIGHTGBM_BACKEND"
log INFO "LightGBM backend: $LIGHTGBM_BACKEND"

CURRENT_STAGE="GPU driver validation"
NVIDIA_SMI=""
if [[ "$ACTION" == "prepare" ]]; then
    log INFO "CPU preparation mode: skipping GPU driver and runtime checks"
else
    NVIDIA_SMI="$(type -P nvidia-smi || true)"
    if [[ -z "$NVIDIA_SMI" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        NVIDIA_SMI=/usr/lib/wsl/lib/nvidia-smi
    fi
    [[ -n "$NVIDIA_SMI" ]] || \
        die "nvidia-smi is missing; install a compatible NVIDIA driver first"
    if (( IS_WSL )) && [[ ! -e /dev/dxg ]]; then
        die "WSL GPU device /dev/dxg is not visible. Run outside this sandbox/container or enable GPU passthrough"
    fi
    if ! "$NVIDIA_SMI" >/dev/null 2>&1; then
        die "nvidia-smi cannot access the GPU; check the driver and container/WSL passthrough"
    fi
    GPU_STATUS="$($NVIDIA_SMI --query-gpu=name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | head -n 1)"
    log OK "NVIDIA GPU (name, total MiB, used MiB, utilization %): $GPU_STATUS"

    if [[ "$LIGHTGBM_BACKEND" == "gpu" ]]; then
        command -v clinfo >/dev/null 2>&1 || \
            die "OpenCL backend requires clinfo and a vendor GPU ICD (Ubuntu loader: sudo apt install clinfo ocl-icd-libopencl1)"
        if ! clinfo 2>/dev/null | grep -Eiq 'Device Type.*GPU|CL_DEVICE_TYPE_GPU'; then
            die "clinfo does not report a physical GPU; a CPU OpenCL runtime is not accepted"
        fi
    fi
fi

if [[ "$LIGHTGBM_BACKEND" == "cuda" ]] && ! command -v g++ >/dev/null 2>&1; then
    die "g++ is required to build LightGBM CUDA (Ubuntu: sudo apt install build-essential)"
fi

download_file() {
    local url="$1"
    local target="$2"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --connect-timeout 20 \
            --output "$target" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=20 --output-document="$target" "$url"
    else
        die "curl or wget is required to download Miniconda"
    fi
}

find_or_install_conda() {
    local candidate
    for candidate in \
        "$(type -P conda || true)" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/miniforge3/bin/conda" \
        "$ROOT_DIR/.tools/miniconda/bin/conda"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            CONDA_BIN="$candidate"
            return 0
        fi
    done

    local tools_dir="$ROOT_DIR/.tools"
    local download_dir="$ROOT_DIR/.downloads"
    local install_prefix="$tools_dir/miniconda"
    local installer="$download_dir/Miniconda3-py310_26.5.3-1-Linux-x86_64.sh"
    local partial="$installer.part"
    local url="https://repo.anaconda.com/miniconda/Miniconda3-py310_26.5.3-1-Linux-x86_64.sh"
    local expected="4a82fe0a50a28e8a9406b3ed8e465b7009aa7d0225566802c3370df96b10d834"

    [[ ! -e "$install_prefix" ]] || \
        die "$install_prefix exists but is not a valid Miniconda installation"
    command -v sha256sum >/dev/null 2>&1 || \
        die "sha256sum is required to verify the Miniconda installer"
    mkdir -p "$tools_dir" "$download_dir"
    log INFO "Conda not found; downloading the pinned Miniconda installer"
    download_file "$url" "$partial"
    local actual
    actual="$(sha256sum "$partial" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || \
        die "Miniconda SHA256 mismatch: expected $expected, got $actual"
    mv "$partial" "$installer"
    bash "$installer" -b -p "$install_prefix"
    [[ -x "$install_prefix/bin/conda" ]] || die "Miniconda installation failed"
    CONDA_BIN="$install_prefix/bin/conda"
}

CURRENT_STAGE="Conda discovery"
CONDA_BIN=""
find_or_install_conda
log OK "Conda: $CONDA_BIN"

CURRENT_STAGE="Python environment"
if [[ -e "$ENV_PREFIX" && ! -x "$ENV_PREFIX/bin/python" ]]; then
    die "$ENV_PREFIX exists but is not a usable environment; move it or set SMTGAZER_ENV_PREFIX"
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    log INFO "Creating isolated Python 3.10 environment at $ENV_PREFIX"
    "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" --override-channels \
        --channel nvidia --channel defaults \
        python=3.10 pip swig cmake ninja
fi

PYTHON="$ENV_PREFIX/bin/python"
if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
    die "$ENV_PREFIX must contain Python 3.10; existing environments are never deleted automatically"
fi
log OK "Python: $($PYTHON --version 2>&1)"

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR"
if (( IS_WSL )); then
    export PATH="$ENV_PREFIX/bin:/usr/lib/wsl/lib:$PATH"
else
    export PATH="$ENV_PREFIX/bin:$PATH"
fi
export CUDAToolkit_ROOT="$ENV_PREFIX"
export CUDACXX="$ENV_PREFIX/bin/nvcc"

ensure_cuda_toolkit() {
    if [[ -x "$ENV_PREFIX/bin/nvcc" ]] && \
        "$ENV_PREFIX/bin/nvcc" --version | grep -q 'release 12\.1' && \
        [[ -f "$ENV_PREFIX/lib/libcudadevrt.a" ]] && \
        [[ -f "$ENV_PREFIX/lib/libcudart_static.a" ]]; then
        log OK "CUDA 12.1 build toolkit is present"
        return 0
    fi

    CURRENT_STAGE="CUDA toolkit installation"
    log INFO "Installing the user-space CUDA 12.1 compiler and headers"
    "$CONDA_BIN" install --yes --prefix "$ENV_PREFIX" --override-channels \
        --channel nvidia --channel defaults \
        cuda-version=12.1 \
        cuda-nvcc=12.1.105 \
        cuda-cudart=12.1.105 \
        cuda-cudart-dev=12.1.105 \
        cuda-cudart-static=12.1.105 \
        cuda-cccl=12.1.109
    "$ENV_PREFIX/bin/nvcc" --version | grep -q 'release 12\.1' || \
        die "CUDA 12.1 compiler validation failed"
    [[ -f "$ENV_PREFIX/lib/libcudadevrt.a" ]] || \
        die "CUDA device runtime library is missing after installation"
    [[ -f "$ENV_PREFIX/lib/libcudart_static.a" ]] || \
        die "CUDA static runtime library is missing after installation"
    log OK "CUDA 12.1 build toolkit installed"
}

if [[ "$LIGHTGBM_BACKEND" == "cuda" ]]; then
    ensure_cuda_toolkit
fi

runtime_ready() {
    (
        cd "$SMT_ROOT"
        "$PYTHON" - <<'PY'
from importlib.metadata import version
from pathlib import Path

expected = {
    "ConfigSpace": "1.2.1",
    "dask": "2024.12.1",
    "dask-jobqueue": "0.9.0",
    "emcee": "3.1.6",
    "joblib": "1.4.2",
    "lightgbm": "4.5.0",
    "more-itertools": "10.5.0",
    "numpy": "1.26.4",
    "psutil": "5.9.8",
    "pyclustering": "0.10.1.2",
    "pynisher": "1.0.10",
    "pyrfr": "0.9.0",
    "pySMT": "0.9.6",
    "PyYAML": "6.0.2",
    "regex": "2024.11.6",
    "scikit-learn": "1.6.1",
    "scipy": "1.13.1",
    "torch": "2.2.0",
    "torch-geometric": "2.5.3",
    "typing_extensions": "4.11.0",
}
for package, wanted in expected.items():
    actual = version(package).split("+", 1)[0]
    if actual != wanted:
        raise SystemExit(f"{package}: expected {wanted}, found {actual}")

import smac
import torch

if torch.version.cuda != "12.1":
    raise SystemExit(f"torch CUDA runtime must be 12.1, found {torch.version.cuda}")
smac_path = Path(smac.__file__).resolve()
expected_root = Path.cwd() / "smac"
if expected_root not in smac_path.parents:
    raise SystemExit(f"not using repository SMAC: {smac_path}")
PY
    )
}

CURRENT_STAGE="Python dependency installation"
CUDA_BUILD_MARKER="$ENV_PREFIX/.smtgazer-lightgbm-cuda-4.5.0"
if runtime_ready >/dev/null 2>&1 && "$PYTHON" -m pip check >/dev/null 2>&1; then
    log OK "Pinned Python runtime is already installed"
else
    log INFO "Installing pinned Python runtime dependencies"
    rm -f "$CUDA_BUILD_MARKER"
    "$PYTHON" -m pip install --disable-pip-version-check \
        pip==26.1.2 setuptools==69.5.1 wheel==0.47.0
    "$PYTHON" -m pip install --disable-pip-version-check \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    "$PYTHON" -m pip install --disable-pip-version-check \
        -r "$SMT_ROOT/requirements-runtime.txt"
    "$PYTHON" -m pip install --disable-pip-version-check --no-deps \
        --config-settings editable_mode=compat -e "$SMT_ROOT/smac"
    runtime_ready
    "$PYTHON" -m pip check
    log OK "Pinned Python runtime installed"
fi

lightgbm_cuda_probe() {
    "$PYTHON" - <<'PY'
import lightgbm as lgb
import numpy as np

rng = np.random.RandomState(0)
features = rng.rand(512, 8)
target = rng.rand(512)
model = lgb.LGBMRegressor(
    n_estimators=2,
    min_data_in_leaf=1,
    max_bin=63,
    device_type="cuda",
    verbose=-1,
)
model.fit(features, target)
prediction = model.predict(features[:8])
if model.booster_.params.get("device_type") != "cuda":
    raise SystemExit("LightGBM did not retain device_type=cuda")
if not np.isfinite(prediction).all():
    raise SystemExit("LightGBM CUDA prediction is not finite")
PY
}

build_lightgbm_cuda() {
    CURRENT_STAGE="LightGBM CUDA build"
    log INFO "Building LightGBM 4.5.0 from source with USE_CUDA=ON"
    CMAKE_BUILD_PARALLEL_LEVEL="${SMTGAZER_BUILD_JOBS:-2}" \
        "$PYTHON" -m pip install --disable-pip-version-check \
        --force-reinstall --no-deps --no-cache-dir --no-binary lightgbm \
        --config-settings=cmake.define.USE_CUDA=ON lightgbm==4.5.0
    touch "$CUDA_BUILD_MARKER"
}

if [[ "$LIGHTGBM_BACKEND" == "cuda" ]]; then
    if [[ "$ACTION" == "prepare" ]]; then
        if [[ -f "$CUDA_BUILD_MARKER" ]]; then
            log OK "Prepared LightGBM CUDA build is already installed"
        else
            build_lightgbm_cuda
            log OK "LightGBM CUDA build completed; runtime validation is deferred to a GPU node"
        fi
    else
        CURRENT_STAGE="LightGBM CUDA validation"
        if lightgbm_cuda_probe >/dev/null 2>&1; then
            touch "$CUDA_BUILD_MARKER"
            log OK "LightGBM CUDA fit/predict passed"
        else
            build_lightgbm_cuda
            if ! lightgbm_cuda_probe; then
                die "LightGBM CUDA build completed but real fit/predict still failed"
            fi
            log OK "LightGBM CUDA source build and fit/predict passed"
        fi
    fi
fi

CURRENT_STAGE="environment snapshot"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$LOG_DIR/pip-freeze.txt"
if [[ -n "$NVIDIA_SMI" ]]; then
    "$NVIDIA_SMI" > "$LOG_DIR/nvidia-smi.txt"
else
    printf 'GPU runtime checks deferred by prepare action.\n' > "$LOG_DIR/nvidia-smi.txt"
fi
log OK "Environment snapshot saved in $LOG_DIR"

if [[ "$ACTION" == "prepare" ]]; then
    log OK "CPU-side environment preparation completed"
    log INFO "Submit a GPU job that runs: ./run_smtgazer.sh run --dataset DATASET --lightgbm-backend $LIGHTGBM_BACKEND"
    exit 0
fi

if [[ "$ACTION" == "setup" ]]; then
    log OK "Environment setup completed"
    log INFO "Next command: ./run_smtgazer.sh check --lightgbm-backend $LIGHTGBM_BACKEND"
    exit 0
fi

CURRENT_STAGE="strict GPU smoke test"
(
    cd "$SMT_ROOT"
    "$PYTHON" -B check_server.py \
        --torch-device "$TORCH_DEVICE" \
        --lightgbm-device-type "$LIGHTGBM_BACKEND"
)

if [[ "$ACTION" == "check" ]]; then
    log OK "All environment and GPU checks passed"
    exit 0
fi

GPU_MONITOR_FILE="$LOG_DIR/gpu.csv"
GPU_SUMMARY_FILE="$LOG_DIR/gpu-summary.json"
GPU_PHASE_FILE="$LOG_DIR/gpu-phase.txt"
export SMTGAZER_GPU_PHASE_FILE="$GPU_PHASE_FILE"
printf 'orchestration\n' > "$GPU_PHASE_FILE"
GPU_INDEX="${TORCH_DEVICE#cuda:}"
if [[ "$GPU_INDEX" == "$TORCH_DEVICE" ]]; then
    GPU_INDEX=0
fi
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "--torch-device must look like cuda:0"

monitor_gpu() {
    printf '%s\n' 'timestamp_utc,phase,index,name,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,power_draw_w,power_limit_w,temperature_c,sm_clock_mhz'
    while true; do
        local timestamp phase sample
        timestamp="$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')"
        phase="$(tr -d '\r\n,' < "$GPU_PHASE_FILE")"
        sample="$("$NVIDIA_SMI" --id="$GPU_INDEX" \
            --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
            --format=csv,noheader,nounits | head -n 1)" || return
        printf '%s,%s,%s\n' "$timestamp" "${phase:-unknown}" "$sample"
        sleep "$GPU_SAMPLE_INTERVAL"
    done
}

monitor_gpu > "$GPU_MONITOR_FILE" &
GPU_MONITOR_PID=$!
cleanup_gpu_monitor() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    if [[ $(wc -l < "$GPU_MONITOR_FILE") -gt 1 ]]; then
        if "$PYTHON" -B "$SMT_ROOT/summarize_gpu_metrics.py" \
            --input "$GPU_MONITOR_FILE" --output "$GPU_SUMMARY_FILE"; then
            log OK "GPU metric summary: $GPU_SUMMARY_FILE"
        else
            log WARN "GPU samples were saved but summary generation failed"
        fi
    fi
}
trap cleanup_gpu_monitor EXIT
log INFO "Continuous GPU monitor: $GPU_MONITOR_FILE"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

common_driver_args() {
    DRIVER_ARGS=(
        "$PYTHON" -B "$SMT_ROOT/run_experiment.py"
        --dataset "$DATASET"
        --device "$TORCH_DEVICE"
    )
    if [[ -n "$CLUSTERS" ]]; then
        DRIVER_ARGS+=(--clusters "$CLUSTERS")
    fi
    if (( OVERWRITE )); then
        DRIVER_ARGS+=(--overwrite)
    fi
    if (( TRAIN_SIBYL )); then
        DRIVER_ARGS+=(--allow-partial-features)
    fi
}

feature_driver_args() {
    common_driver_args
    DRIVER_ARGS+=(
        --query-root "$QUERY_ROOT"
        --sibyl-root "$SIBYL_ROOT"
    )
    if (( TRAIN_SIBYL )); then
        DRIVER_ARGS+=(
            --train-sibyl
            --gnn-epochs "$GNN_EPOCHS"
            --gnn-batch-size "$GNN_BATCH_SIZE"
            --gnn-workers "$GNN_WORKERS"
            --graph-workers "$GRAPH_WORKERS"
            --feature-batch-size "$FEATURE_BATCH_SIZE"
            --gnn-seed "$GNN_SEED"
        )
    else
        DRIVER_ARGS+=(--checkpoint "$CHECKPOINT")
    fi
}

run_driver() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    (
        cd "$SMT_ROOT"
        "$@"
    )
}

if (( OVERWRITE )); then
    log WARN "--overwrite is active; canonical feature/result files are not backed up"
fi

if [[ "$RUN_STAGE" == "full" || "$RUN_STAGE" == "features" ]]; then
    CURRENT_STAGE="feature generation"
    feature_driver_args
    DRIVER_ARGS+=(--stage features --seeds "${SEEDS[@]}")
    run_driver "${DRIVER_ARGS[@]}"
fi

run_portfolio_batch() {
    local -a batch=("$@")
    common_driver_args
    DRIVER_ARGS+=(--stage portfolio --seeds "${batch[@]}")
    run_driver "${DRIVER_ARGS[@]}"
}

if [[ "$RUN_STAGE" == "full" || "$RUN_STAGE" == "portfolio" ]]; then
    CURRENT_STAGE="portfolio optimization"
    HAS_ZERO=0
    REMAINING_SEEDS=()
    for seed in "${SEEDS[@]}"; do
        if [[ "$seed" == "0" ]]; then
            HAS_ZERO=1
        else
            REMAINING_SEEDS+=("$seed")
        fi
    done
    if (( HAS_ZERO )) && (( ${#SEEDS[@]} > 1 )); then
        run_portfolio_batch 0
        run_portfolio_batch "${REMAINING_SEEDS[@]}"
    else
        run_portfolio_batch "${SEEDS[@]}"
    fi
fi

log OK "Experiment completed"
log INFO "Results: $SMT_ROOT/output"
log INFO "Full log: $LOG_FILE"
