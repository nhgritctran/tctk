import datetime
import os
import subprocess
import sys
import time


class Dsub:
    """
    This class is a wrapper to run dsub on the All of Us researcher workbench.
    Params input_dict and output_dict values must be paths to Google Cloud Storage bucket(s).
    """

    def __init__(
        self,
        docker_image: str,
        job_script_name: str,
        job_name: str,
        input_dict: {},
        output_dict: {},
        env_dict: {},
        log_file_path=None,
        machine_type: str = "c3d-highcpu-4",
        disk_type="pd-ssd",
        boot_disk_size=50,
        disk_size=256,
        user_project=os.getenv("GOOGLE_PROJECT"),
        project=os.getenv("GOOGLE_PROJECT"),
        dsub_user_name=None,
        user_name=None,
        bucket=os.getenv("WORKSPACE_BUCKET"),
        google_project=os.getenv("GOOGLE_PROJECT"),
        region="us-central1",
        provider="google-batch",
        preemptible=False,
        use_private_address: bool = True,
        custom_args=None,
        use_aou_docker_prefix: bool = True,
    ):
        # Resolve defaults that depend on environment variables
        if dsub_user_name is None:
            _email = os.getenv("OWNER_EMAIL", "")
            dsub_user_name = _email.split("@")[0] if _email else ""
        if user_name is None:
            _email = os.getenv("OWNER_EMAIL", "")
            user_name = _email.split("@")[0].replace(".", "-") if _email else ""

        # Standard attributes
        self.docker_image = docker_image
        self.job_script_name = job_script_name
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.env_dict = env_dict
        self.machine_type = machine_type
        self.disk_type = disk_type
        self.boot_disk_size = boot_disk_size
        self.disk_size = disk_size
        self.user_project = user_project
        self.project = project
        self.dsub_user_name = dsub_user_name
        self.user_name = user_name
        self.bucket = bucket
        self.job_name = job_name.replace("_", "-")
        self.google_project = google_project
        self.region = region
        self.provider = provider
        self.preemptible = preemptible
        self.use_private_address = use_private_address
        self.custom_args = custom_args
        self.use_aou_docker_prefix = use_aou_docker_prefix

        # Internal attributes for optional naming conventions
        self.date = datetime.date.today().strftime("%Y%m%d")
        self.time = datetime.datetime.now().strftime("%H%M%S")

        # log file path
        if log_file_path is not None:
            self.log_file_path = log_file_path
        else:
            self.log_file_path = (
                f"{self.bucket}/dsub/logs/{self.job_name}/{self.user_name}/{self.date}/{self.time}/{self.job_name}.log"
            )

        # some reporting attributes
        self.script = ""
        self.dsub_command = ""
        self.job_id = ""
        self.job_stdout = self.log_file_path.replace(".log", "-stdout.log")
        self.job_stderr = self.log_file_path.replace(".log", "-stderr.log")
        self.dsub_start_time = None
        self.dsub_end_time = None
        self.dsub_runtime = None

    def dsub_base_script(self) -> str:
        """
        Generate the base dsub command script with core configuration parameters.

        This method extracts the base dsub command structure without input/output
        flags, environment variables, or script commands. Useful for creating
        custom dsub commands or testing.

        :return: Base dsub command with provider, machine, networking, and logging configuration
        :rtype: str
        """
        if self.use_aou_docker_prefix:
            aou_docker_prefix = os.getenv("ARTIFACT_REGISTRY_DOCKER_REPO")
            image_tag = f"{aou_docker_prefix}/{self.docker_image}"
        else:
            image_tag = self.docker_image

        base_script = (
            f"dsub" + " " +
            f"--provider \"{self.provider}\"" + " " +
            f"--regions \"{self.region}\"" + " " +
            f"--machine-type \"{self.machine_type}\"" + " " +
            f"--boot-disk-size {self.boot_disk_size}" + " " +
            f"--disk-size {self.disk_size}" + " " +
            f"--user-project \"{self.user_project}\"" + " " +
            f"--project \"{self.project}\"" + " " +
            f"--image \"{image_tag}\"" + " " +
            f"--network \"global/networks/network\"" + " " +
            f"--subnetwork \"regions/{self.region}/subnetworks/subnetwork\"" + " " +
            f"--service-account \"$(gcloud config get-value account)\"" + " " +
            f"--user \"{self.dsub_user_name}\"" + " " +
            f"--logging {self.log_file_path} $@" + " " +
            f"--name \"{self.job_name}\"" + " " +
            f"--env GOOGLE_PROJECT=\"{self.google_project}\"" + " "
        )

        return base_script

    @staticmethod
    def _merge_custom_args(base_command: str, custom_args: str) -> str:
        """
        Merge custom arguments with base command, allowing custom args to override existing ones.

        :param base_command: The base command string
        :type base_command: str
        :param custom_args: Custom arguments that may override existing ones
        :type custom_args: str
        :return: Merged command string with custom args taking precedence
        :rtype: str
        """
        import re

        if not custom_args:
            return base_command

        # Parse custom args to find argument names
        custom_arg_pattern = r'--([a-zA-Z-]+)(?:\s+[^\s-]+|\s*=\s*[^\s]+)?'
        custom_matches = re.findall(custom_arg_pattern, custom_args)
        custom_arg_names = set(custom_matches)

        # Remove conflicting arguments from base command
        modified_command = base_command
        for arg_name in custom_arg_names:
            # Pattern to match the argument and its value
            # Handles both --arg value and --arg=value formats
            patterns = [
                rf'--{re.escape(arg_name)}\s+(?:"[^"]*"|\'[^\']*\'|\S+)',  # --arg value
                rf'--{re.escape(arg_name)}=(?:"[^"]*"|\'[^\']*\'|\S+)',    # --arg=value
                rf'--{re.escape(arg_name)}(?=\s|$)'                        # --arg (flag only)
            ]

            for pattern in patterns:
                modified_command = re.sub(pattern, '', modified_command)

        # Clean up extra spaces
        modified_command = re.sub(r'\s+', ' ', modified_command).strip()

        # Add custom arguments
        return modified_command + " " + custom_args

    def _dsub_script(self):
        """
        Generate the dsub command script with all configured parameters.

        :return: Complete dsub command as a string
        :rtype: str
        """
        # Get base script
        base_script = self.dsub_base_script()

        # add disk-type
        disk_type_flag = ""
        if self.disk_type is not None:
            disk_type_flag = f"--disk-type \"{self.disk_type}\"" + " "

        # generate input flags
        input_flags = ""
        if len(self.input_dict) > 0:
            for k, v in self.input_dict.items():
                input_flags += f"--input {k}={v}" + " "

        # generate output flag
        output_flags = ""
        if len(self.output_dict) > 0:
            for k, v in self.output_dict.items():
                output_flags += f"--output {k}={v}" + " "

        # generate env flags
        env_flags = ""
        if len(self.env_dict) > 0:
            for k, v in self.env_dict.items():
                env_flags += f"--env {k}=\"{v}\"" + " "

        # job script flag
        job_script = f"--script {self.job_script_name}" + " "

        # combined script
        script = base_script + disk_type_flag + env_flags + input_flags + output_flags + job_script

        # add preemptible argument if used
        if self.preemptible:
            script += " --preemptible"

        # add use-private-address if used
        if self.use_private_address:
            script += " --use-private-address"

        # merge custom arguments with potential overrides
        if self.custom_args is not None:
            script = self._merge_custom_args(script, self.custom_args)

        # add attribute for convenience
        self.script = script

        return script

    def _check_job_status(self, stdout: str):
        """
        Check job status from dstat output and identify terminal states.

        :param stdout: Output from dstat command
        :return: Tuple of (status_value, has_success, has_failed, has_canceled, last_update, status_detail)
        """
        status_value = ""
        has_success = False
        has_failed = False
        has_canceled = False
        last_update = ""
        status_detail = ""

        if stdout:
            # Look for status, status-detail, and last-update lines
            for line in stdout.split('\n'):
                line_stripped = line.strip().lower()
                original_line = line.strip()

                if line_stripped.startswith('status:'):
                    # Extract everything after 'status:' and clean it up
                    status_part = line_stripped.split('status:', 1)[1].strip()
                    status_value = status_part.rstrip('.,!?;:')
                elif line_stripped.startswith('status-detail:'):
                    # Extract status detail
                    status_detail = original_line.split(':', 1)[1].strip()
                elif line_stripped.startswith('last-update:'):
                    # Extract last update timestamp and remove microseconds and quotes
                    last_update = original_line.split(':', 1)[1].strip()
                    last_update = last_update.strip("'")  # Remove single quotes
                    if '.' in last_update:
                        last_update = last_update.split('.')[0]

            if status_value:
                # Define status patterns
                success_patterns = ["success", "succeeded", "complete", "completed", "finished", "done"]
                failed_patterns = ["unsuccessful", "incomplete", "failed", "error", "failure", "timeout"]
                canceled_patterns = ["aborted", "terminated", "cancelled", "canceled", "delete", "deleted"]

                # Check status against patterns
                has_success = any(pattern in status_value for pattern in success_patterns)
                has_failed = any(pattern in status_value for pattern in failed_patterns)
                has_canceled = any(pattern in status_value for pattern in canceled_patterns)

        return status_value, has_success, has_failed, has_canceled, last_update, status_detail

    def check_status(
        self,
        full: bool = False,
        custom_args: str = None,
        streaming: bool = False,
        update_interval: int = 10,
        verbose: bool = False,
        auto_job_cleanup: bool = False,
        cleanup_delay: int = 0
    ):
        """
        Check the status of the submitted job using dstat command.

        :param full: Whether to show full detailed status information
        :param custom_args: Additional custom arguments for dstat command
        :param streaming: Whether to continuously monitor status with auto-refresh
        :param update_interval: Seconds between status updates when streaming
        :param verbose: Whether to print debug information for status detection
        :param auto_job_cleanup: Whether to automatically cleanup job after completion or failure
        :param cleanup_delay: Seconds to wait after completion/failure before cleaning up job
        """

        # base command
        check_status = (
            f"dstat --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\" --status \"*\""
        )

        # full static status
        if full:
            check_status += " --full"

        # merge custom arguments with potential overrides
        if custom_args is not None:
            check_status = self._merge_custom_args(check_status, custom_args)

        if streaming:
            # Auto-detect notebook
            try:
                # noinspection PyUnresolvedReferences
                from IPython.display import clear_output
            except ImportError:
                pass

            last_status = ""
            last_check_time = 0

            # Print initial runtime line
            print(f"Refresh interval: {update_interval}s | Runtime: Initializing...", end="", flush=True)

            while True:
                current_time = time.time()

                # Calculate runtime based on dsub job start time
                if self.dsub_start_time is not None:
                    runtime = datetime.datetime.now() - self.dsub_start_time
                    runtime_str = str(runtime).split('.')[0]  # Remove microseconds
                else:
                    runtime_str = "Unknown (job not started)"

                # Check status only at specified intervals
                if current_time - last_check_time >= update_interval:
                    last_check_time = current_time

                    # Run command and capture output
                    result = subprocess.run([check_status], shell=True, capture_output=True, text=True)
                    current_status = result.stdout.strip()

                    # Check for terminal states using full status
                    status_value = ""
                    last_update = ""
                    status_detail = ""
                    if result.stdout:
                        # Get full status to check actual job status line
                        full_status_cmd = check_status + " --full"
                        full_result = subprocess.run([full_status_cmd], shell=True, capture_output=True, text=True)

                        # Use helper function to check job status
                        status_value, has_success, has_failed, has_canceled, last_update, status_detail = self._check_job_status(full_result.stdout)

                        if verbose:
                            print(f"\nDEBUG - Status value: '{status_value}'")
                            print(
                                f"DEBUG - has_success: {has_success}, has_failed: {has_failed}, has_canceled: {has_canceled}")
                            print()

                        if has_success:
                            self.dsub_end_time = datetime.datetime.now()
                            if self.dsub_start_time is not None:
                                self.dsub_runtime = self.dsub_end_time - self.dsub_start_time
                            print("\r" + " " * 80)  # Clear current line
                            print(f"\rLast update: {last_update}")
                            print(f"Job status: {status_value.upper()}")
                            if status_detail:
                                print(f"Status detail: {status_detail}")
                            print()
                            print("Job completed successfully!")
                            print()

                            # Auto-cleanup job if requested
                            if auto_job_cleanup:
                                if cleanup_delay > 0:
                                    print()
                                    for i in range(cleanup_delay, 0, -1):
                                        print(f"\rCleaning up job in {i} seconds...", end="", flush=True)
                                        time.sleep(1)
                                    print("\r" + " " * 40, end="")
                                    print("\rCleaning up job...")
                                else:
                                    print("Cleaning up job...")
                                self.kill()
                            break

                        # Check for failure patterns
                        if has_failed:
                            self.dsub_end_time = datetime.datetime.now()
                            if self.dsub_start_time is not None:
                                self.dsub_runtime = self.dsub_end_time - self.dsub_start_time
                            print("\r" + " " * 80)  # Clear current line
                            print(f"\rLast update: {last_update}")
                            print(f"Job status: {status_value.upper()}")
                            if status_detail:
                                print(f"Status detail: {status_detail}")
                            print()
                            print("Job failed!")
                            print()

                            # Print job logs when it fails
                            print()
                            print("FINAL LOGS:")
                            print()
                            try:
                                print("=== FULL STATUS ===")
                                print()
                                print(full_result.stdout)
                                print("===== STDOUT ======")
                                print()
                                self.view_log("stdout", n_lines=50)
                                print()
                                print("===== STDERR ======")
                                print()
                                self.view_log("stderr", n_lines=50)
                                print()
                            except Exception as e:
                                print(f"Could not retrieve logs: {e}")
                                print()

                            # Auto-cleanup job if requested
                            if auto_job_cleanup:
                                if cleanup_delay > 0:
                                    print()
                                    for i in range(cleanup_delay, 0, -1):
                                        print(f"\rCleaning up job in {i} seconds...", end="", flush=True)
                                        time.sleep(1)
                                    print("\r" + " " * 40, end="")
                                    print("\rCleaning up job...")
                                else:
                                    print("Cleaning up job...")
                                self.kill()
                            break

                        # Check for canceled/deleted patterns
                        if has_canceled:
                            self.dsub_end_time = datetime.datetime.now()
                            if self.dsub_start_time is not None:
                                self.dsub_runtime = self.dsub_end_time - self.dsub_start_time
                            print("\r" + " " * 80)  # Clear current line
                            print(f"\rLast update: {last_update}")
                            print(f"Job status: {status_value.upper()}")
                            if status_detail:
                                print(f"Status detail: {status_detail}")
                            print()
                            print("Job was canceled or deleted!")
                            break

                    # Check for empty status (worker shutdown)
                    if not current_status and self.dsub_start_time is not None and (datetime.datetime.now() - self.dsub_start_time).total_seconds() > 60:
                        print("\r" + " " * 80)  # Clear current line
                        print("\rNo job status found - worker has likely shut down")
                        break

                    # Check if status changed (use last_update + status_detail for comparison)
                    current_formatted = f"{last_update}|{status_detail}"
                    if current_formatted != last_status:
                        # Clear current runtime line and replace with new status
                        print("\r" + " " * 80)  # Clear current line
                        if last_update:
                            print(f"\r{last_update}")
                        print(f"Job Status: {status_value.upper()}")
                        if status_detail:
                            print(f"Status Detail: {status_detail}")
                        print()
                        last_status = current_formatted
                        # Print new runtime line below the status
                        print(f"Refresh interval: {update_interval}s | Runtime: {runtime_str}", end="", flush=True)
                    else:
                        # Update runtime line in place
                        print(f"\rRefresh interval: {update_interval}s | Runtime: {runtime_str}", end="", flush=True)
                else:
                    # Just update runtime display every second
                    print(f"\rRefresh interval: {update_interval}s | Runtime: {runtime_str}", end="", flush=True)

                # Sleep for 1 second to create smooth runtime updates
                time.sleep(1)
        else:
            # Run status check once
            result = subprocess.run([check_status], shell=True, capture_output=True, text=True)
            print(result.stdout)

            # For auto-cleanup in non-streaming mode, check if job is completed/failed
            if auto_job_cleanup and result.stdout:
                # Get full status to check actual job status
                full_status_cmd = check_status + " --full"
                full_result = subprocess.run([full_status_cmd], shell=True, capture_output=True, text=True)

                # Use helper function to check job status
                status_value, has_success, has_failed, has_canceled, last_update, status_detail = self._check_job_status(full_result.stdout)

                if has_success or has_failed:
                    print()
                    if has_success:
                        print("Job completed successfully!")
                    else:
                        print("Job failed!")
                    if cleanup_delay > 0:
                        print()
                        for i in range(cleanup_delay, 0, -1):
                            print(f"\rCleaning up job in {i} seconds...", end="", flush=True)
                            time.sleep(1)
                        print("\r" + " " * 40, end="")
                        print("\rCleaning up job...")
                    else:
                        print("Cleaning up job...")
                    self.kill()

    def view_log(self, log_type="stdout", n_lines=10):

        tail = f" | head -n {n_lines}"

        if log_type == "stdout":
            full_command = f"gsutil cat {self.job_stdout}" + tail
        elif log_type == "stderr":
            full_command = f"gsutil cat {self.job_stderr}" + tail
        elif log_type == "full":
            full_command = f"gsutil cat {self.log_file_path}" + tail
        else:
            print("log_type must be 'stdout', 'stderr', or 'full'.")
            sys.exit(1)

        subprocess.run([full_command], shell=True)

    def kill(self):
        """
        Kill/cancel the running job using ddel command.

        Note: Requires that the job has been submitted and job_id is available.
        """
        kill_job = (
            f"ddel --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\""
        )
        subprocess.run([kill_job], shell=True)

    def view_all(self):
        """
        View all running jobs linked to user account and project using dstat command.
        """
        view_jobs = (
            f"dstat --provider {self.provider} --users \"{self.user_name}\" --project {self.project} --jobs \"*\" "
        )
        subprocess.run([view_jobs], shell=True)

    def kill_all(self):
        """
        Kill/cancel all running jobs linked to user account and project using ddel command.
        """
        kill_jobs = (
            f"ddel --provider {self.provider} --users \"{self.user_name}\" --project {self.project} --jobs \"*\" "
        )
        subprocess.run([kill_jobs], shell=True)

    def echo_hello_test(
        self,
        stream_status: bool = True,
        update_interval: int = 5,
        use_private_address: bool = True,
        custom_args: str = None
    ):
        """
        Run a simple echo test using dsub to verify configuration and connectivity.

        This method executes a minimal dsub job that echoes "Hello" to stdout.
        Useful for testing dsub setup, authentication, and basic job execution
        without running complex scripts.

        :param stream_status: Whether to automatically stream status after submission
        :type stream_status: bool
        :param update_interval: Seconds between status updates when streaming
        :type update_interval: int
        :param use_private_address: Whether to use private IP addresses
        :type use_private_address: bool
        :param custom_args: Additional custom arguments for dsub command
        :type custom_args: str | None
        """
        # Get base script and add command
        test_command = self.dsub_base_script() + " --command 'echo Hello'"

        # Add use-private-address if specified
        if use_private_address:
            test_command += " --use-private-address"

        # merge custom arguments with potential overrides
        if custom_args is not None:
            test_command = self._merge_custom_args(test_command, custom_args)

        print(f"Running echo test for job: {self.job_name}")
        print("Test command: echo Hello")
        print()

        # Print the dsub command with spaces replaced by \n for readability
        print("dsub command:")
        formatted_command = test_command.replace("--", "\\ \n--")
        print(formatted_command)
        print()

        # Execute the test command
        process = subprocess.Popen(
            test_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Echo test submitted successfully!")
            self.job_id = stdout.strip()
            self.dsub_start_time = datetime.datetime.now()  # Record job start time
            print("job-id:", stdout)
            print()

            if stream_status:
                print("#" * 80)
                print()
                print("Starting status monitoring...")
                print()
                self.check_status(streaming=True, update_interval=update_interval, auto_job_cleanup=True, cleanup_delay=10)
            else:
                print("To check status: dsub_instance.check_status()")
                print("To view output: dsub_instance.view_log('stdout')")
        else:
            print("Echo test submission failed!")
            print("Error:", stderr)

    def run(self, show_command=False):

        s = subprocess.run([self._dsub_script()], shell=True, capture_output=True, text=True)

        if s.returncode == 0:
            print(f"Successfully run dsub to schedule job {self.job_name}.")
            self.job_id = s.stdout.strip()
            self.dsub_start_time = datetime.datetime.now()  # Record job start time
            print("job-id:", s.stdout)
            print()
            self.dsub_command = s.args[0].replace("--", "\\ \n--")
            if show_command:
                print("dsub command:")
                print(self.dsub_command)

        else:
            print(f"Failed to run dsub to schedule job {self.job_name}.")
            print()
            print("Error information:")
            print(s.stderr)
            self.dsub_command = s.args[0].replace("--", "\\ \n--")
            if show_command:
                print("dsub command:")
                print(self.dsub_command)