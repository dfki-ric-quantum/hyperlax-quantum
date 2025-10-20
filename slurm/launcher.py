import argparse
import logging
import os
import re
import shlex
import subprocess
import uuid

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PATH_OF_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
# Assumes this script is in `slurm/`, so go up one level to get the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(PATH_OF_THIS_FILE, os.pardir))


# --- Reusable Utility Functions ---
def get_git_tag() -> str:
    """Gets the current git tag or commit hash for tracking."""
    try:
        git_tag = subprocess.check_output(["git", "describe", "--tags", "--always"], stderr=subprocess.PIPE).decode("utf-8").strip()
        return git_tag
    except subprocess.CalledProcessError:
        try:
            count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], stderr=subprocess.PIPE).decode("utf-8").strip()
            hash_val = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.PIPE).decode("utf-8").strip()
            return f"no-tag-{count}-g{hash_val}"
        except subprocess.CalledProcessError:
            return "unknown"

def run_sbatch_command(command: str) -> str:
    """Submits a command to SLURM via sbatch and returns the job ID."""
    logger.info(f"Submitting sbatch command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"sbatch command failed with error: {e.stderr}")
        raise

def main(args):
    # --- WANDB Tags (Generic and useful for any job) ---
    tags = []
    if args.auto_tag:
        git_tag = get_git_tag()
        if git_tag:
            tags.append(git_tag)
    if args.job_name:
        tags.append(args.job_name)
    if args.additional_tags:
        tags.extend(args.additional_tags)

    if tags:
        # We set WANDB_TAGS so the script running inside Singularity inherits it.
        os.environ["WANDB_TAGS"] = ",".join(tags)
        logger.info(f"Setting WANDB_TAGS for submitted job: {os.environ['WANDB_TAGS']}")

    # --- Load SLURM Template ---
    try:
        with open(args.slurm_template_path) as f:
            slurm_template = f.read()
    except OSError as e:
        logger.error(f"Failed to read SLURM template '{args.slurm_template_path}': {e}")
        return

    # --- Path and Job Name Generation ---
    job_uuid = uuid.uuid4().hex[:8]

    # Sanitize job name for use in paths and SLURM names
    sanitized_job_name = re.sub(r'[^a-z0-9_]', '', args.job_name.lower().replace("-", "_"))
    job_name_for_slurm = f"{sanitized_job_name}_{job_uuid}"

    # Centralized directory for all SLURM log files
    slurm_log_base_dir = os.path.join(args.output_root_base, "slurm_logs")
    slurm_log_path_for_job = os.path.join(slurm_log_base_dir, job_name_for_slurm)
    os.makedirs(slurm_log_path_for_job, exist_ok=True)

    # --- Assemble the Command ---
    command_args = args.command
    # The user might use '--' to separate launcher args from the command.
    # argparse.REMAINDER captures it, so we must remove it if present.
    if command_args and command_args[0] == '--':
        command_args = command_args[1:]

    # Use shlex.join to create a single, properly quoted command string.
    # This correctly handles arguments with spaces or special characters.
    user_command_str = shlex.join(command_args)
    logger.info(f"Final command to be executed in job: {user_command_str}")

    if args.exclude:
        exclude_nodes = ','.join(args.exclude)
        exclude_directive = f"#SBATCH --exclude={exclude_nodes}"
    else:
        exclude_directive = ""

    # #TODO
    # if args.exclusive:
    #     exclusive_directive = "#SBATCH --exclusive"
    # else:
    #     exclusive_directive = ""


    # --- Fill SLURM Template ---
    replacements = {
        "{{job_name_suffix}}": job_name_for_slurm,
        "{{slurmlogpath}}": slurm_log_path_for_job,
        "{{time}}": args.time,
        "{{gres}}": args.gres,
        "{{exclude_directive}}": exclude_directive,
        #"{{exclusive_directive}}": exclusive_directive,
        "{{singularity_image_path}}": args.singularity_image,
        "{{host_project_root}}": PROJECT_ROOT,
        "{{command}}": user_command_str,
    }

    job_script = slurm_template
    for key, value in replacements.items():
        job_script = job_script.replace(key, str(value))

    # --- Save and Submit SLURM Job Script ---
    slurm_script_filename = f"job_{job_name_for_slurm}.slurm"
    slurm_script_path = os.path.join(slurm_log_path_for_job, slurm_script_filename)

    try:
        with open(slurm_script_path, "w") as f:
            f.write(job_script)
        logger.info(f"SLURM job script saved to: {slurm_script_path}")
    except OSError as e:
        logger.error(f"Failed to write SLURM job script: {e}")
        return

    try:
        job_id = run_sbatch_command(f"sbatch --parsable {slurm_script_path}")
        logger.info("\n--- JOB SUBMITTED SUCCESSFULLY ---")
        logger.info(f"Job Name: {job_name_for_slurm}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"SLURM output will be in: {slurm_log_path_for_job}/slurm-{job_id}.out")
        logger.info(f"Any artifacts from your script should be in: {args.output_root_base}")
        logger.info("------------------------------------")
    except Exception as e:
        logger.error(f"Failed to submit SLURM job: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A generic SLURM job launcher for running any command within a Singularity container.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Job Definition ---
    parser.add_argument("--job-name", type=str, required=True, help="A descriptive name for the job (e.g., 'runtime_benchmark').")

    # --- SLURM Configuration ---
    parser.add_argument("--time", required=True, help="Time limit for the SLURM job (e.g., '0-01:00:00').")
    parser.add_argument("--gres", required=True, help="GPU resources (e.g., 'gpu:1', 'gpu:rtx8000:2').")
    parser.add_argument("--slurm-template-path", default="slurm/gpu.slurm_template", help="Path to the SLURM template file.")
    parser.add_argument("--singularity-image", required=True, help="Absolute path to the Singularity .sif image file on the cluster nodes.")
    parser.add_argument("--output-root-base", required=True, help="Base directory where all SLURM logs and job artifacts will be saved.")
    parser.add_argument("--exclude", nargs="+", help="List of SLURM nodes to exclude.")

    # --- Optional Features ---
    parser.add_argument("--auto-tag", action="store_true", help="Enable auto-tagging with Git info.")
    parser.add_argument("--additional-tags", nargs="+", help="Additional custom tags for W&B.")

    # --- The Command to Execute ---
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command and its arguments to execute inside the container.\n"
             "IMPORTANT: This must be the LAST argument. Use '--' to be safe.\n"
             "Example: ... -- python my_script.py --arg1 value1"
    )

    args = parser.parse_args()

    if not args.command or (len(args.command) == 1 and args.command[0] == '--'):
        parser.error("The 'command' argument is required. Please provide the command to execute after all other options.")

    main(args)
