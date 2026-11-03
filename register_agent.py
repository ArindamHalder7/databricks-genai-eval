#!/usr/bin/env python
"""Register the QA agent (agent/qa_agent_model.py) as a new Unity Catalog model version.

Runs from GitHub Actions (or any machine / Databricks job) whenever the agent code changes.
Creating the new version automatically triggers the model's MLflow deployment job, which
evaluates that version and opens a Jira bug if the evaluation fails.

Authentication: standard Databricks SDK env vars, e.g. DATABRICKS_HOST + DATABRICKS_TOKEN
(or DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET for a service principal).
The Anthropic key is read from ANTHROPIC_API_KEY, or else from the Databricks secret
<scope>/<key> (default anthropic/api_key), so no extra GitHub secret is required.

Examples:
    python register_agent.py
    python register_agent.py --skip-if-unchanged --wait-for-eval
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "accenture.default.genai_qa_agent"
DEFAULT_EXPERIMENT = "/Users/arindam.halder@accenture.com/Agent Registry CICD"
PIP_REQUIREMENTS = ["anthropic>=0.38.0,<0.50.0"]
INPUT_EXAMPLE = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default=os.getenv("AGENT_MODEL_NAME", DEFAULT_MODEL), help="UC model catalog.schema.name")
    p.add_argument("--agent-file", default=os.path.join(HERE, "agent", "qa_agent_model.py"), help="code-based MLflow model file")
    p.add_argument("--experiment", default=os.getenv("AGENT_EXPERIMENT", DEFAULT_EXPERIMENT), help="MLflow experiment path")
    p.add_argument("--secret-scope", default="anthropic")
    p.add_argument("--secret-key", default="api_key")
    p.add_argument("--git-sha", default=os.getenv("GITHUB_SHA"))
    p.add_argument("--git-branch", default=os.getenv("GITHUB_REF_NAME"))
    p.add_argument("--git-repo", default=os.getenv("GITHUB_REPOSITORY"))
    p.add_argument("--skip-if-unchanged", action="store_true", help="skip when code hash equals the latest version's")
    p.add_argument("--wait-for-eval", action="store_true", help="wait for the deployment job run and exit 1 if it fails")
    p.add_argument("--eval-timeout", type=int, default=1800, help="seconds to wait for the evaluation")
    return p.parse_args()


def git_value(args_value, *cmd):
    if args_value:
        return args_value
    try:
        return subprocess.check_output(["git", *cmd], cwd=HERE, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def agent_code_version(path):
    m = re.search(r'^AGENT_CODE_VERSION\s*=\s*["\']([^"\']+)', open(path, encoding="utf-8").read(), re.M)
    return m.group(1) if m else "unknown"


def ensure_anthropic_key(scope, key):
    """log_model loads the agent; make the key available without storing it in Git."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    secret = WorkspaceClient().secrets.get_secret(scope=scope, key=key)
    os.environ["ANTHROPIC_API_KEY"] = base64.b64decode(secret.value).decode("utf-8")


def latest_version(client, model_name):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    # UC search results do not include tags; fetch the full model version
    return client.get_model_version(model_name, latest.version)


def write_github_output(values):
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            for k, v in values.items():
                f.write(f"{k}={v}\n")


def wait_for_deployment_eval(model_name, version, timeout_s):
    """Find the deployment job run for this version and wait for it. Returns (result_state, run_url)."""
    w = WorkspaceClient()
    job_id = getattr(MlflowClient().get_registered_model(model_name), "deployment_job_id", None)
    if not job_id:
        print("No deployment job connected to the model; nothing to wait for.")
        return None, None
    deadline, run = time.time() + timeout_s, None
    while time.time() < deadline:
        if run is None:
            for r in w.jobs.list_runs(job_id=int(job_id), limit=25):
                params = {p.name: p.value for p in (r.job_parameters or [])}
                if params.get("model_name") == model_name and str(params.get("model_version")) == str(version):
                    run = r
                    print(f"Deployment job run found: {r.run_page_url}")
                    break
        if run is not None:
            run = w.jobs.get_run(run.run_id)
            life = run.state.life_cycle_state.value if run.state and run.state.life_cycle_state else None
            if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                return (run.state.result_state.value if run.state.result_state else life), run.run_page_url
        time.sleep(20)
    return "TIMEOUT", run.run_page_url if run else None


def main():
    args = parse_args()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)
    client = MlflowClient()

    code_hash = file_sha256(args.agent_file)
    git_sha = git_value(args.git_sha, "rev-parse", "HEAD")
    git_branch = git_value(args.git_branch, "rev-parse", "--abbrev-ref", "HEAD")
    git_repo = args.git_repo or "local"

    prev = latest_version(client, args.model_name)
    if args.skip_if_unchanged and prev is not None and (prev.tags or {}).get("code_sha256") == code_hash:
        result = {"status": "SKIPPED", "reason": "agent code unchanged", "model_name": args.model_name,
                  "model_version": prev.version}
        print(json.dumps(result))
        write_github_output({"status": "SKIPPED", "model_version": prev.version})
        return 0

    ensure_anthropic_key(args.secret_scope, args.secret_key)
    tags = {"git_sha": git_sha, "git_branch": git_branch, "git_repo": git_repo, "code_sha256": code_hash,
            "agent_code_version": agent_code_version(args.agent_file), "registered_by": "register_agent.py"}

    with mlflow.start_run(run_name=f"register-{git_sha[:8]}") as run:
        mlflow.set_tags(tags)
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=args.agent_file,
            pip_requirements=PIP_REQUIREMENTS,
            registered_model_name=args.model_name,  # new version -> deployment job auto-triggers
            input_example=INPUT_EXAMPLE,
        )
    version = str(info.registered_model_version)
    for k, v in tags.items():
        client.set_model_version_tag(args.model_name, version, k, v)

    result = {"status": "REGISTERED", "model_name": args.model_name, "model_version": version,
              "previous_version": prev.version if prev else None, "mlflow_run_id": run.info.run_id,
              "model_uri": f"models:/{args.model_name}/{version}", **tags}
    print(json.dumps(result, indent=2))
    write_github_output({"status": "REGISTERED", "model_version": version})

    if args.wait_for_eval:
        state, url = wait_for_deployment_eval(args.model_name, version, args.eval_timeout)
        print(f"Deployment evaluation: {state} {url or ''}")
        write_github_output({"eval_state": state, "eval_run_url": url or ""})
        if state not in (None, "SUCCESS"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
