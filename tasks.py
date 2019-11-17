from pathlib import Path

from invoke import task

PROJECT_DIR = Path.cwd()
WORKDIR = Path("/app")


def _volume(relative_path):
    return f"--volume {PROJECT_DIR / relative_path}:{WORKDIR / relative_path}"


@task
def gpu(context, model_args=""):
    volumes = ["data", "logs"]
    volumes_option_string = " ".join((_volume(v) for v in volumes))
    context.run(
        f"docker run "
        f"{volumes_option_string} "
        f"--gpus all "
        f"mutusfa/invoicenet "
        f"python3 {model_args}"
    )
    print(volumes_option_string)
