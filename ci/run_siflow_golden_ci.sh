#!/usr/bin/env bash

set -euo pipefail

: "${CI_ROOT:?CI_ROOT is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

CI_COMMIT_DIR="${CI_ROOT}/${GITHUB_SHA}"
TEST_ENVIRONMENT="${TEST_ENVIRONMENT:-dev}"
TEST_PLATFORM="${TEST_PLATFORM:-dgx_h100}"
DATA_PATH="${DATA_PATH:-/volume/hisys}"
TRAINING_SCRIPT_PATH="${TRAINING_SCRIPT_PATH:-pretrain_gpt.py}"
N_REPEAT="${N_REPEAT:-1}"
ENABLE_LIGHTWEIGHT_MODE="${ENABLE_LIGHTWEIGHT_MODE:-false}"
RECORD_CHECKPOINTS="${RECORD_CHECKPOINTS:-false}"
COUNT_PER_POD="${COUNT_PER_POD:-8}"

export SLURM_NODEID="${SLURM_NODEID:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${COUNT_PER_POD}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/usr}"

ensure_yq() {
    if [[ -x /usr/local/bin/yq ]]; then
        /usr/local/bin/yq --version
        return
    fi

    mkdir -p /usr/local/bin

    if command -v yq >/dev/null 2>&1; then
        ln -sf "$(command -v yq)" /usr/local/bin/yq 2>/dev/null || cp "$(command -v yq)" /usr/local/bin/yq
        chmod +x /usr/local/bin/yq
        /usr/local/bin/yq --version
        return
    fi

    case "$(uname -m)" in
        x86_64|amd64) yq_arch="amd64" ;;
        aarch64|arm64) yq_arch="arm64" ;;
        *)
            echo "Unsupported architecture for yq install: $(uname -m)"
            exit 1
            ;;
    esac

    yq_version="${YQ_VERSION:-v4.44.3}"
    yq_url="https://github.com/mikefarah/yq/releases/download/${yq_version}/yq_linux_${yq_arch}"
    yq_tmp="$(mktemp)"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${yq_url}" -o "${yq_tmp}"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "${yq_tmp}" "${yq_url}"
    else
        python3 - "${yq_url}" "${yq_tmp}" <<'PY'
import sys
import urllib.request

url, path = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as response, open(path, "wb") as output:
    output.write(response.read())
PY
    fi

    install -m 0755 "${yq_tmp}" /usr/local/bin/yq
    rm -f "${yq_tmp}"
    /usr/local/bin/yq --version
}

apt_install() {
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get is not available; cannot install missing system package(s): $*"
        return 1
    fi

    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

ensure_pybind11() {
    if python3 -m pybind11 --includes >/dev/null 2>&1; then
        python3 -m pybind11 --includes
        return
    fi

    echo "Missing python pybind11 module; installing it for dataset helper build."

    if command -v uv >/dev/null 2>&1 && uv pip install --system pybind11; then
        :
    else
        if ! python3 -m pip --version >/dev/null 2>&1; then
            apt_install python3-pip
        fi
        python3 -m pip install --break-system-packages pybind11 || python3 -m pip install pybind11
    fi

    if ! python3 -m pybind11 --includes >/dev/null 2>&1; then
        apt_install python3-pip
        python3 -m pip install --break-system-packages pybind11
    fi

    python3 -m pybind11 --includes
}

ensure_python_headers() {
    python_include_dir="$(
        python3 - <<'PY'
import sysconfig

print(sysconfig.get_paths()["include"])
PY
    )"

    if [[ -f "${python_include_dir}/Python.h" ]]; then
        echo "Python headers: ${python_include_dir}"
        return
    fi

    python_dev_package="$(
        python3 - <<'PY'
import sys

print(f"python{sys.version_info.major}.{sys.version_info.minor}-dev")
PY
    )"

    echo "Missing Python.h under ${python_include_dir}; installing ${python_dev_package}."
    apt_install "${python_dev_package}" || apt_install python3-dev

    if [[ ! -f "${python_include_dir}/Python.h" ]]; then
        echo "Python headers are still missing after install: ${python_include_dir}/Python.h"
        exit 1
    fi

    echo "Python headers: ${python_include_dir}"
}

ensure_dataset_helper_build_deps() {
    ensure_python_headers
    ensure_pybind11
}

ensure_yq
ensure_dataset_helper_build_deps

if [[ -n "${TEST_SPECS:-}" ]]; then
    IFS=';' read -r -a test_specs <<< "${TEST_SPECS}"
else
    : "${TEST_MODEL:?TEST_MODEL is required when TEST_SPECS is empty}"
    : "${TEST_CASE:?TEST_CASE is required when TEST_SPECS is empty}"
    test_specs=("${TEST_MODEL}:${TEST_CASE}")
fi

mkdir -p "${CI_COMMIT_DIR}"

echo "Megatron golden CI"
echo "Commit: ${GITHUB_SHA}"
echo "CI root: ${CI_COMMIT_DIR}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Tests: ${test_specs[*]}"

for spec in "${test_specs[@]}"; do
    IFS=':' read -r test_model test_case test_script <<< "${spec}"
    test_script="${test_script:-${TRAINING_SCRIPT_PATH}}"

    if [[ -z "${test_model}" || -z "${test_case}" ]]; then
        echo "Invalid TEST_SPECS entry: ${spec}"
        echo "Expected format: model:test_case[:training_script]"
        exit 1
    fi

    test_name="${test_model}/${test_case}"
    output_path="${CI_COMMIT_DIR}/outputs/${test_name}/${TEST_ENVIRONMENT}_${TEST_PLATFORM}"
    tensorboard_path="${output_path}/tensorboard"
    checkpoint_save_path="${CI_COMMIT_DIR}/checkpoints/${test_name}"
    data_cache_path="${CI_COMMIT_DIR}/data-cache/${test_name}"
    tmp_checkpoint_path="${CI_COMMIT_DIR}/tmp/checkpoints/${test_name}"
    checkpoint_load_path="${CHECKPOINT_LOAD_PATH:-${CI_COMMIT_DIR}/checkpoint-load/${test_name}}"
    training_params_path="./tests/functional_tests/test_cases/${test_name}/model_config.yaml"
    golden_values_path="./tests/functional_tests/test_cases/${test_name}/golden_values_${TEST_ENVIRONMENT}_${TEST_PLATFORM}.json"

    echo
    echo "==> Running ${test_name}"
    echo "Model config: ${training_params_path}"
    echo "Golden values: ${golden_values_path}"
    echo "Output path: ${output_path}"

    test -f "${training_params_path}" || { echo "Missing model config: ${training_params_path}"; exit 1; }
    test -f "${golden_values_path}" || { echo "Missing golden values: ${golden_values_path}"; exit 1; }

    rm -rf "${output_path}" "${checkpoint_save_path}" "${data_cache_path}" "${tmp_checkpoint_path}"
    mkdir -p \
        "${output_path}" \
        "${tensorboard_path}" \
        "${checkpoint_save_path}" \
        "${data_cache_path}" \
        "${tmp_checkpoint_path}"
    mkdir -p "${checkpoint_load_path}" || true

    rm -rf /tmp/checkpoints
    ln -s "${tmp_checkpoint_path}" /tmp/checkpoints

    arguments=(
        "DATA_PATH=${DATA_PATH}"
        "DATA_CACHE_PATH=${data_cache_path}"
        "OUTPUT_PATH=${output_path}"
        "TENSORBOARD_PATH=${tensorboard_path}"
        "CHECKPOINT_SAVE_PATH=${checkpoint_save_path}"
        "CHECKPOINT_LOAD_PATH=${checkpoint_load_path}"
        "TRAINING_SCRIPT_PATH=${test_script}"
        "TRAINING_PARAMS_PATH=${training_params_path}"
        "GOLDEN_VALUES_PATH=${golden_values_path}"
        "N_REPEAT=${N_REPEAT}"
        "ENABLE_LIGHTWEIGHT_MODE=${ENABLE_LIGHTWEIGHT_MODE}"
        "RECORD_CHECKPOINTS=${RECORD_CHECKPOINTS}"
    )

    bash ./tests/functional_tests/shell_test_utils/run_ci_test.sh "${arguments[@]}"
done

echo
echo "All golden-value tests passed."
