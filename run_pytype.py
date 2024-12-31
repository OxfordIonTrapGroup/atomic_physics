import os
import subprocess


def main():
    if os.name != "posix":
        raise RuntimeError(
            "pyptype is not supported on Windows, "
            "but you should be able to run it in the WSL."
        )

    subprocess.check_call("poetry run pytype -k .".split())
