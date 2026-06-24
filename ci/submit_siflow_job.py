import argparse
import json
import os
import time
from pathlib import Path


DEFAULT_TEST_SPECS = (
    "gpt:gpt3_mcore_te_tp1_pp1_dist_optimizer_commoncrawl_deepseek_tokenizer;"
    "moe:gpt3_moe_mcore_te_tp2_pp1_ep2_dist_optimizer_commoncrawl_deepseek_tokenizer"
)

DEFAULT_CMD = r"""
set -euo pipefail

CI_COMMIT_DIR="${CI_ROOT}/${GITHUB_SHA}"
WORKDIR="${CI_COMMIT_DIR}/repo"
CI_STATUS_DIR="${CI_COMMIT_DIR}/status"
CI_STATUS_FILE="${CI_STATUS_DIR}/result.json"

write_ci_status() {
    local state="$1"
    local exit_code="${2:-null}"
    local updated_at
    updated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    python3 - "${CI_STATUS_FILE}" "${state}" "${exit_code}" "${updated_at}" <<'PY'
import json
import os
import sys

path, state, exit_code, updated_at = sys.argv[1:5]
payload = {
    "state": state,
    "updated_at": updated_at,
    "repository": os.environ.get("GITHUB_REPOSITORY", ""),
    "ref": os.environ.get("GITHUB_REF_NAME", ""),
    "sha": os.environ.get("GITHUB_SHA", ""),
}
payload["exit_code"] = None if exit_code == "null" else int(exit_code)

tmp_path = f"{path}.tmp"
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True)
    f.write("\n")
os.replace(tmp_path, path)
PY
}

finish_ci_status() {
    local exit_code="$?"
    trap - EXIT

    if [[ "${exit_code}" -eq 0 ]]; then
        write_ci_status success 0
    else
        write_ci_status failure "${exit_code}"
    fi

    exit "${exit_code}"
}

rm -rf "${WORKDIR}"
mkdir -p "${CI_COMMIT_DIR}" "${CI_STATUS_DIR}"
write_ci_status running null
trap finish_ci_status EXIT

git clone "https://x-access-token:${REPO_CLONE_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "${WORKDIR}"
cd "${WORKDIR}"
if [[ -n "${GITHUB_REF:-}" ]]; then
    git fetch origin "${GITHUB_REF}" || git fetch origin "${GITHUB_SHA}"
else
    git fetch origin "${GITHUB_SHA}"
fi
git checkout --detach "${GITHUB_SHA}"
git remote set-url origin "https://github.com/${GITHUB_REPOSITORY}.git"

echo "Checked out ${GITHUB_REPOSITORY}@${GITHUB_SHA}"

bash ./ci/run_siflow_golden_ci.sh
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Submit a SiFlow training task.")
    parser.add_argument("--exp-name", default="ci-test", help="Task name prefix.")
    parser.add_argument("--nodes", type=int, default=1, help="Total nodes, including master.")
    parser.add_argument("--region", default="ap-southeast", help="SiFlow region.")
    parser.add_argument("--cluster", default="aries", help="SiFlow cluster.")
    parser.add_argument("--resource-pool", default="ls-cpt", help="Resource pool.")
    parser.add_argument("--instance", default="sci.g20-3", help="Instance type.")
    parser.add_argument("--count-per-pod", type=int, default=8, help="Device count per pod.")
    parser.add_argument("--image", default="miles", help="Image name.")
    parser.add_argument("--image-version", default="deepseek-v4", help="Image version.")
    parser.add_argument("--image-url", default="radixark/miles:deepseek-v4", help="Image URL.")
    parser.add_argument("--image-type", default="radixark", help="Image type.")
    parser.add_argument("--priority", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--cmd", default=DEFAULT_CMD, help="Command to run in the task.")
    parser.add_argument("--repo", default="", help="GitHub repository name.")
    parser.add_argument("--ref", default="", help="Git ref name.")
    parser.add_argument("--github-ref", default="", help="Full GitHub ref, such as refs/pull/1/merge.")
    parser.add_argument("--sha", default="", help="Git commit SHA.")
    parser.add_argument("--github-token-env", default="REPO_CLONE_TOKEN")
    parser.add_argument("--ci-root", default="/volume/hisys/ci/Megatron-LM")
    parser.add_argument(
        "--test-specs",
        default=DEFAULT_TEST_SPECS,
        help="Semicolon-separated model:test_case[:training_script] entries.",
    )
    parser.add_argument("--test-model", default="gpt")
    parser.add_argument(
        "--test-case", default="gpt3_mcore_te_tp1_pp1_dist_optimizer_commoncrawl_deepseek_tokenizer"
    )
    parser.add_argument("--test-environment", default="dev")
    parser.add_argument("--test-platform", default="dgx_h100")
    parser.add_argument("--data-path", default="/volume/hisys")
    parser.add_argument("--checkpoint-load-path", default="")
    parser.add_argument("--training-script-path", default="pretrain_gpt.py")
    parser.add_argument("--n-repeat", default="1")
    parser.add_argument("--enable-lightweight-mode", default="false")
    parser.add_argument("--record-checkpoints", default="false")
    parser.add_argument("--wait", action="store_true", help="Block until the SiFlow task finishes.")
    parser.add_argument(
        "--poll-interval", type=int, default=60, help="Seconds between SiFlow status polls."
    )
    parser.add_argument(
        "--wait-timeout", type=int, default=21600, help="Maximum seconds to wait for task finish."
    )
    parser.add_argument(
        "--result-file-grace",
        type=int,
        default=300,
        help="Seconds to wait for the shared result file after SiFlow reports task success.",
    )
    parser.add_argument(
        "--no-require-result-file",
        action="store_false",
        dest="require_result_file",
        help="Allow API success to pass even if the shared CI result file is missing.",
    )
    parser.set_defaults(require_result_file=True)

    args = parser.parse_args()
    if args.nodes < 1:
        parser.error("--nodes must be >= 1")
    if args.count_per_pod < 1:
        parser.error("--count-per-pod must be >= 1")
    if args.poll_interval < 1:
        parser.error("--poll-interval must be >= 1")
    if args.wait_timeout < 1:
        parser.error("--wait-timeout must be >= 1")
    if args.result_file_grace < 0:
        parser.error("--result-file-grace must be >= 0")

    return args


SUCCESS_STATUSES = {"complete", "completed", "finished", "success", "succeeded"}
FAILURE_STATUSES = {
    "cancelled",
    "canceled",
    "error",
    "failed",
    "failure",
    "killed",
    "stopped",
    "terminated",
    "timeout",
    "timedout",
}


def normalize_status(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()

    for attr in ("phase", "status", "state", "name", "value"):
        if hasattr(value, attr):
            normalized = normalize_status(getattr(value, attr))
            if normalized:
                return normalized

    return str(value).strip().lower()


def get_field(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def get_task_status(task):
    for source in (task, get_field(task, "status")):
        for name in ("status", "phase", "state"):
            status = normalize_status(get_field(source, name))
            if status:
                return status
    return "unknown"


def get_task_message(task):
    for source in (task, get_field(task, "status")):
        for name in ("message", "msg", "status_msg", "statusMessage", "reason"):
            message = get_field(source, name)
            if message:
                return str(message)
    return ""


def read_result_file(result_path):
    try:
        with result_path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"[wait] Result file exists but is not valid JSON yet: {exc}", flush=True)
        return None


def result_path_for(args):
    if not args.sha:
        return None
    return Path(args.ci_root) / args.sha / "status" / "result.json"


def wait_for_task(client, uuid, args):
    result_path = result_path_for(args)
    deadline = time.monotonic() + args.wait_timeout
    last_status = None
    last_result_state = None
    success_without_result_since = None

    print(f"[wait] Blocking until SiFlow task finishes: {uuid}", flush=True)
    if result_path is not None:
        print(f"[wait] Result file: {result_path}", flush=True)

    while True:
        if result_path is not None:
            result = read_result_file(result_path)
            if result is not None:
                result_state = normalize_status(result.get("state"))
                exit_code = result.get("exit_code")
                if result_state != last_result_state:
                    print(
                        f"[wait] Result file state={result_state or 'unknown'} "
                        f"exit_code={exit_code}",
                        flush=True,
                    )
                    last_result_state = result_state

                if result_state in SUCCESS_STATUSES or exit_code == 0:
                    print("[wait] Remote CI result succeeded.", flush=True)
                    return 0
                if result_state in FAILURE_STATUSES or (
                    isinstance(exit_code, int) and exit_code != 0
                ):
                    print("[wait] Remote CI result failed.", flush=True)
                    return 1

        try:
            task = client.tasks.get(uuid=uuid)
            status = get_task_status(task)
            message = get_task_message(task)
        except Exception as exc:
            status = "poll_error"
            message = str(exc)

        if status != last_status:
            suffix = f" message={message}" if message else ""
            print(f"[wait] SiFlow status={status}{suffix}", flush=True)
            last_status = status

        if status in FAILURE_STATUSES:
            print("[wait] SiFlow task failed before a successful CI result was written.", flush=True)
            return 1

        if status in SUCCESS_STATUSES:
            if result_path is None or not args.require_result_file:
                print("[wait] SiFlow task succeeded.", flush=True)
                return 0
            now = time.monotonic()
            if success_without_result_since is None:
                success_without_result_since = now
                print(
                    "[wait] SiFlow reports success; waiting for required CI result file.",
                    flush=True,
                )
            elif now - success_without_result_since >= args.result_file_grace:
                print(
                    "[wait] SiFlow task finished, but no final CI result file was written.",
                    flush=True,
                )
                return 1
        else:
            success_without_result_since = None

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"[wait] Timed out after {args.wait_timeout}s waiting for {uuid}.", flush=True)
            return 124

        time.sleep(min(args.poll_interval, remaining))


def submit_train(args):
    from siflow import SiFlow
    from siflow.types import TaskEnv, TaskUserSelectedInstance, TaskVolume

    client = SiFlow(
        region=args.region,
        cluster=args.cluster,
        access_key_id=os.environ["SCITIX_AK"],
        access_key_secret=os.environ["SCITIX_SK"],
    )

    uuid = client.tasks.create(
        name_prefix=args.exp_name,
        image=args.image,
        image_version=args.image_version,
        image_url=args.image_url,
        image_type=args.image_type,
        type="pytorchjob",
        priority=args.priority,
        cmd=args.cmd,
        workers=args.nodes - 1,
        resource_pool=args.resource_pool,
        instances=[
            TaskUserSelectedInstance(name=args.instance, count_per_pod=args.count_per_pod),
        ],
        volumes=[
            TaskVolume(mount_dir="/volume/data", volume_name='ai4s-data'),
            TaskVolume(mount_dir="/volume/code", volume_name='ai4s-code'),
            TaskVolume(mount_dir="/volume/hisys", volume_name='hisys-data'),
        ],
        task_env=[
            TaskEnv(env_key="NUM_NODES", env_value=str(args.nodes), hide=False),
            TaskEnv(env_key="GITHUB_REPOSITORY", env_value=args.repo, hide=False),
            TaskEnv(env_key="GITHUB_REF_NAME", env_value=args.ref, hide=False),
            TaskEnv(env_key="GITHUB_REF", env_value=args.github_ref, hide=False),
            TaskEnv(env_key="GITHUB_SHA", env_value=args.sha, hide=False),
            TaskEnv(env_key="GITHUB_SERVER_URL", env_value="https://github.com", hide=False),
            TaskEnv(env_key="REPO_CLONE_TOKEN", env_value=os.environ[args.github_token_env], hide=True),
            TaskEnv(env_key="CI_ROOT", env_value=args.ci_root, hide=False),
            TaskEnv(env_key="TEST_SPECS", env_value=args.test_specs, hide=False),
            TaskEnv(env_key="TEST_MODEL", env_value=args.test_model, hide=False),
            TaskEnv(env_key="TEST_CASE", env_value=args.test_case, hide=False),
            TaskEnv(env_key="TEST_ENVIRONMENT", env_value=args.test_environment, hide=False),
            TaskEnv(env_key="TEST_PLATFORM", env_value=args.test_platform, hide=False),
            TaskEnv(env_key="DATA_PATH", env_value=args.data_path, hide=False),
            TaskEnv(env_key="CHECKPOINT_LOAD_PATH", env_value=args.checkpoint_load_path, hide=False),
            TaskEnv(env_key="TRAINING_SCRIPT_PATH", env_value=args.training_script_path, hide=False),
            TaskEnv(env_key="N_REPEAT", env_value=args.n_repeat, hide=False),
            TaskEnv(
                env_key="ENABLE_LIGHTWEIGHT_MODE",
                env_value=args.enable_lightweight_mode,
                hide=False,
            ),
            TaskEnv(env_key="RECORD_CHECKPOINTS", env_value=args.record_checkpoints, hide=False),
            TaskEnv(env_key="COUNT_PER_POD", env_value=str(args.count_per_pod), hide=False),
        ],
    )
    print(f"[train] UUID: {uuid}")
    return client, uuid


def main():
    args = parse_args()
    client, uuid = submit_train(args)
    if args.wait:
        return wait_for_task(client, uuid, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
