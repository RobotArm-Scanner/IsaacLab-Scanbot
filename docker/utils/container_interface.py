# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .state_file import StateFile


class ContainerInterface:
    """A helper class for managing Isaac Lab containers."""

    @staticmethod
    def _strip_quotes(value: str) -> str:
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1].strip()
        return value

    @staticmethod
    def _normalize_suffix(value: str) -> str:
        """Normalize a suffix to either "" or a string starting with "-"."""
        value = ContainerInterface._strip_quotes(value)
        if not value:
            return ""
        if value.startswith("-"):
            return value
        return f"-{value}"

    def __init__(
        self,
        context_dir: Path,
        profile: str = "base",
        yamls: list[str] | None = None,
        envs: list[str] | None = None,
        statefile: StateFile | None = None,
        suffix: str | None = None,
    ):
        """Initialize the container interface with the given parameters.

        Args:
            context_dir: The context directory for Docker operations.
            profile: The profile name for the container. Defaults to "base".
            yamls: A list of yaml files to extend ``docker-compose.yaml`` settings. These are extended in the order
                they are provided.
            envs: A list of environment variable files to extend the ``.env.base`` file. These are extended in the order
                they are provided.
            statefile: An instance of the :class:`Statefile` class to manage state variables. Defaults to None, in
                which case a new configuration object is created by reading the configuration file at the path
                ``context_dir/.container.cfg``.
            suffix: Optional docker image and container name suffix. Defaults to None, in which case the suffix is
                loaded from the composed env files (``DOCKER_NAME_SUFFIX``) and falls back to the empty string. A
                leading hyphen is inserted if missing. For example, if "base" is passed to profile, and "custom" is
                passed to suffix, then the produced docker image and container will be named ``isaac-lab-base-custom``.
        """
        # set the context directory
        self.context_dir = context_dir

        # create a state-file if not provided
        # the state file is a manager of run-time state variables that are saved to a file
        if statefile is None:
            self.statefile = StateFile(path=self.context_dir / ".container.cfg")
        else:
            self.statefile = statefile

        # set the profile and container name
        self.profile = profile
        if self.profile == "isaaclab":
            # Silently correct from isaaclab to base, because isaaclab is a commonly passed arg
            # but not a real profile
            self.profile = "base"

        # keep track of extra env files passed by user to include for any profile
        self.extra_envs = envs or []

        # resolve the image extension through the passed yamls and envs
        self._resolve_image_extension(yamls, envs)
        # load the environment variables from the .env files
        self._parse_dot_vars()

        # set the docker image and container name suffix:
        # - CLI --suffix wins (expects "dev", becomes "-dev")
        # - otherwise, if PROJECT_SUFFIX is defined in env files, derive suffix from it
        # - otherwise, fall back to DOCKER_NAME_SUFFIX from env files (accepts "-dev" or "dev")
        if suffix is not None and suffix != "":
            self.suffix = self._normalize_suffix(suffix)
        elif "PROJECT_SUFFIX" in self.dot_vars:
            project_suffix = self._strip_quotes(str(self.dot_vars.get("PROJECT_SUFFIX", "")))
            if project_suffix.startswith("-"):
                project_suffix = project_suffix[1:]
            self.suffix = f"-{project_suffix}" if project_suffix else ""
        else:
            env_suffix = str(self.dot_vars.get("DOCKER_NAME_SUFFIX", ""))
            self.suffix = self._normalize_suffix(env_suffix)

        self.container_name = f"isaac-lab-{self.profile}{self.suffix}"
        self.image_name = f"isaac-lab-{self.profile}{self.suffix}:latest"

        # keep the environment variables from the current environment,
        # except make sure that the docker name suffix is set from the script
        self.environ = os.environ.copy()
        self.environ["DOCKER_NAME_SUFFIX"] = self.suffix

    """
    Operations.
    """

    def is_container_running(self) -> bool:
        """Check if the container is running.

        Returns:
            True if the container is running, otherwise False.
        """
        status = subprocess.run(
            ["docker", "container", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return status == "running"

    def does_image_exist(self) -> bool:
        """Check if the Docker image exists.

        Returns:
            True if the image exists, otherwise False.
        """
        result = subprocess.run(["docker", "image", "inspect", self.image_name], capture_output=True, text=True)
        return result.returncode == 0

    def start(self, skip_build: bool = False):
        """Build and start the Docker container using the Docker compose command.

        Args:
            skip_build: If True, do not build images prior to starting the container.
        """
        action = "Starting" if skip_build else "Building and starting"
        print(
            f"[INFO] {action} the container '{self.container_name}' in the background...\n"
        )
        # Check if the container history file exists
        container_history_file = self.context_dir / ".isaac-lab-docker-history"
        if not container_history_file.exists():
            # Create the file with sticky bit on the group
            container_history_file.touch(mode=0o2644, exist_ok=True)

        # build the image(s) for the base profile if not running base (unless skipped)
        if not skip_build and self.profile != "base":
            self.build(targets=["base"], no_cache=False)

        # build the image for the profile
        up_args = ["up", "--detach", "--remove-orphans"]
        if not skip_build:
            up_args.insert(2, "--build")
        subprocess.run(
            ["docker", "compose"]
            + self.add_yamls
            + self.add_profiles
            + self.add_env_files
            + up_args,
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )

    def build(self, targets: list[str] | None = None, no_cache: bool = False):
        """Build selected images using Docker compose.

        Args:
            targets: List of profile names to build (e.g., base, ros2, scanbot). If None, defaults to
                ["base"] when profile is "base", else ["base", profile].
            no_cache: When True, build with --no-cache.
        """
        if not targets:
            targets = ["base"] if self.profile == "base" else ["base", self.profile]

        # Ensure deterministic order: build base first when included
        if "base" in targets:
            targets = ["base"] + [t for t in targets if t != "base"]

        for target_profile in targets:
            services = self._services_for_profile(target_profile)
            if not services:
                raise RuntimeError(f"No services found for profile '{target_profile}'.")
            cmd = [
                "docker",
                "compose",
                *self.add_yamls,
                *self._env_files_args_for_profile(target_profile),
                "--profile",
                target_profile,
                "build",
            ]
            if no_cache:
                cmd.append("--no-cache")
            cmd += services
            subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)

    def _env_files_args_for_profile(self, profile: str) -> list[str]:
        args: list[str] = ["--env-file", ".env.base"]
        env_profile = self.context_dir / f".env.{profile}"
        if env_profile.exists():
            args += ["--env-file", f".env.{profile}"]
        for env in self.extra_envs:
            args += ["--env-file", env]
        return args

    def _services_for_profile(self, profile: str) -> list[str]:
        """Return list of service names for the given profile by querying compose config."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                *self.add_yamls,
                *self._env_files_args_for_profile(profile),
                "--profile",
                profile,
                "config",
                "--services",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )
        services = [s.strip() for s in result.stdout.splitlines() if s.strip()]
        return services

    def enter(self):
        """Enter the running container by executing a bash shell.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Entering the existing '{self.container_name}' container in a bash session...\n")
            subprocess.run([
                "docker",
                "exec",
                "--interactive",
                "--tty",
                *(["-e", f"DISPLAY={os.environ['DISPLAY']}"] if "DISPLAY" in os.environ else []),
                f"{self.container_name}",
                "bash",
            ])
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def stop(self):
        """Stop the running container using the Docker compose command.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Stopping the launched docker container '{self.container_name}'...\n")
            subprocess.run(
                ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["down", "--volumes"],
                check=False,
                cwd=self.context_dir,
                env=self.environ,
            )
        else:
            raise RuntimeError(f"Can't stop container '{self.container_name}' as it is not running.")

    def copy(self, output_dir: Path | None = None):
        """Copy artifacts from the running container to the host machine.

        Args:
            output_dir: The directory to copy the artifacts to. Defaults to None, in which case
                the context directory is used.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Copying artifacts from the '{self.container_name}' container...\n")
            if output_dir is None:
                output_dir = self.context_dir

            # create a directory to store the artifacts
            output_dir = output_dir.joinpath("artifacts")
            if not output_dir.is_dir():
                output_dir.mkdir()

            # define dictionary of mapping from docker container path to host machine path
            docker_isaac_lab_path = Path(self.dot_vars["DOCKER_ISAACLAB_PATH"])
            artifacts = {
                docker_isaac_lab_path.joinpath("logs"): output_dir.joinpath("logs"),
                docker_isaac_lab_path.joinpath("docs/_build"): output_dir.joinpath("docs"),
                docker_isaac_lab_path.joinpath("data_storage"): output_dir.joinpath("data_storage"),
            }
            # print the artifacts to be copied
            for container_path, host_path in artifacts.items():
                print(f"\t -{container_path} -> {host_path}")
            # remove the existing artifacts
            for path in artifacts.values():
                shutil.rmtree(path, ignore_errors=True)

            # copy the artifacts
            for container_path, host_path in artifacts.items():
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"isaac-lab-{self.profile}{self.suffix}:{container_path}/",
                        f"{host_path}",
                    ],
                    check=False,
                )
            print("\n[INFO] Finished copying the artifacts from the container.")
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def config(self, output_yaml: Path | None = None):
        """Process the Docker compose configuration based on the passed yamls and environment files.

        If the :attr:`output_yaml` is not None, the configuration is written to the file. Otherwise, it is printed to
        the terminal.

        Args:
            output_yaml: The path to the yaml file where the configuration is written to. Defaults
                to None, in which case the configuration is printed to the terminal.
        """
        print("[INFO] Configuring the passed options into a yaml...\n")

        # resolve the output argument
        if output_yaml is not None:
            output = ["--output", output_yaml]
        else:
            output = []

        # run the docker compose config command to generate the configuration
        subprocess.run(
            ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["config"] + output,
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )

    """
    Helper functions.
    """

    def _resolve_image_extension(self, yamls: list[str] | None = None, envs: list[str] | None = None):
        """
        Resolve the image extension by setting up YAML files, profiles, and environment files for the Docker compose command.

        Args:
            yamls: A list of yaml files to extend ``docker-compose.yaml`` settings. These are extended in the order
                they are provided.
            envs: A list of environment variable files to extend the ``.env.base`` file. These are extended in the order
                they are provided.
        """
        self.add_yamls = ["--file", "docker-compose.yaml"]
        self.add_profiles = ["--profile", f"{self.profile}"]
        self.add_env_files = ["--env-file", ".env.base"]

        # extend env file based on profile
        if self.profile != "base":
            self.add_env_files += ["--env-file", f".env.{self.profile}"]

        # extend the env file based on the passed envs
        if envs is not None:
            for env in envs:
                self.add_env_files += ["--env-file", env]

        # extend the docker-compose.yaml based on the passed yamls
        if yamls is not None:
            for yaml in yamls:
                self.add_yamls += ["--file", yaml]

    def _parse_dot_vars(self):
        """Parse the environment variables from the .env files.

        Based on the passed ".env" files, this function reads the environment variables and stores them in a dictionary.
        The environment variables are read in order and overwritten if there are name conflicts, mimicking the behavior
        of Docker compose.
        """
        self.dot_vars: dict[str, Any] = {}

        # check if the number of arguments is even for the env files
        if len(self.add_env_files) % 2 != 0:
            raise RuntimeError(
                "The parameters for env files are configured incorrectly. There should be an even number of arguments."
                f" Received: {self.add_env_files}."
            )

        # read the environment variables from the .env files
        for i in range(1, len(self.add_env_files), 2):
            with open(self.context_dir / self.add_env_files[i]) as f:
                self.dot_vars.update(dict(line.strip().split("=", 1) for line in f if "=" in line))
