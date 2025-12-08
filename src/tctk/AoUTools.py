from google.cloud import bigquery
from tableone import TableOne

import datetime
import os
import polars as pl
import subprocess
import sys
import pickle
import tctk.PolarsTools as PT
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
        dsub_user_name=os.getenv("OWNER_EMAIL").split("@")[0],
        user_name=os.getenv("OWNER_EMAIL").split("@")[0].replace(".", "-"),
        bucket=os.getenv("WORKSPACE_BUCKET"),
        google_project=os.getenv("GOOGLE_PROJECT"),
        region="us-central1",
        provider="google-cls-v2",
        preemptible=False,
        use_private_address: bool = True,
        custom_args=None,
        use_aou_docker_prefix: bool = True,
    ):
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


class SocioEconomicStatus:

    def __init__(self, cdr, question_id_dict=None):
        self.cdr = cdr

        self.aou_ses = self.polar_gbq(f"SELECT * FROM {self.cdr}.ds_zip_code_socioeconomic")

        if not question_id_dict:
            self.question_id_dict = {"own_or_rent": 1585370,
                                     "education": 1585940,
                                     "employment_status": 1585952,
                                     "annual_household_income": 1585375}

        self.income_dict = {"Annual Income: less 10k": 1,
                            "Annual Income: 10k 25k": 2,
                            "Annual Income: 25k 35k": 3,
                            "Annual Income: 35k 50k": 4,
                            "Annual Income: 50k 75k": 5,
                            "Annual Income: 75k 100k": 6,
                            "Annual Income: 100k 150k": 7,
                            "Annual Income: 150k 200k": 8,
                            "Annual Income: more 200k": 9}
        self.edu_dict = {"Highest Grade: Never Attended": 1,
                         "Highest Grade: One Through Four": 2,
                         "Highest Grade: Five Through Eight": 3,
                         "Highest Grade: Nine Through Eleven": 4,
                         "Highest Grade: Twelve Or GED": 5,
                         "Highest Grade: College One to Three": 6,
                         "Highest Grade: College Graduate": 7,
                         "Highest Grade: Advanced Degree": 8}
        self.home_dict = {"Current Home Own: Own": "home_own",
                          "Current Home Own: Rent": "home_rent"}
        # "Current Home Own: Other Arrangement" are those with zero in both above categories
        self.employment_dict = {"Employment Status: Employed For Wages": "employed",
                                "Employment Status: Homemaker": "homemaker",
                                "Employment Status: Out Of Work Less Than One": "unemployed_less_1yr",
                                "Employment Status: Out Of Work One Or More": "unemployed_more_1yr",
                                "Employment Status: Retired": "retired",
                                "Employment Status: Self Employed": "self_employed",
                                "Employment Status: Student": "student"}
        # "Employment Status: Unable To Work" are those with zero in all other categories
        self.smoking_dict = {"Smoke Frequency: Every Day": "smoking_every_day",
                             "Smoke Frequency: Some Days": "smoking_some_days"}
        # "Not At All" are those with zero in all other categories

    @staticmethod
    def polar_gbq(query):
        """
        :param query: BigQuery query
        :return: polars dataframe
        """
        client = bigquery.Client()
        query_job = client.query(query)
        rows = query_job.result()
        df = pl.from_arrow(rows.to_arrow())

        return df

    @staticmethod
    def dummy_coding(data, col_name, lookup_dict):
        """
        create dummy variables for a categorical variable
        :param data: polars dataframe
        :param col_name: variable of interest
        :param lookup_dict: dict to map dummy variables
        :return: polars dataframe with new dummy columns
        """
        for k, v in lookup_dict.items():
            data = data.with_columns(pl.when(pl.col(col_name) == k)
                                     .then(1)
                                     .otherwise(0)
                                     .alias(v))

        return data

    def compare_with_median_income(self, data):
        """
        convert area median income to equivalent income bracket and then compare with participant's income bracket
        :param data:
        :return:
        """
        ses_data = self.aou_ses[["PERSON_ID", "ZIP3_AS_STRING", "MEDIAN_INCOME"]]

        # convert zip3 strings to 3 digit codes
        ses_data = ses_data.with_columns(pl.col("ZIP3_AS_STRING").str.slice(0, 3).alias("zip3"))
        ses_data = ses_data.drop("ZIP3_AS_STRING")

        # mapping median income to income brackets
        ses_data = ses_data.with_columns(pl.when((pl.col("MEDIAN_INCOME") >= 0.00) &
                                                 (pl.col("MEDIAN_INCOME") <= 9999.99))
                                         .then(1)
                                         .when((pl.col("MEDIAN_INCOME") >= 10000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 24999.99))
                                         .then(2)
                                         .when((pl.col("MEDIAN_INCOME") >= 25000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 34999.99))
                                         .then(3)
                                         .when((pl.col("MEDIAN_INCOME") >= 35000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 49999.99))
                                         .then(4)
                                         .when((pl.col("MEDIAN_INCOME") >= 50000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 74999.99))
                                         .then(5)
                                         .when((pl.col("MEDIAN_INCOME") >= 75000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 99999.99))
                                         .then(6)
                                         .when((pl.col("MEDIAN_INCOME") >= 100000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 149999.99))
                                         .then(7)
                                         .when((pl.col("MEDIAN_INCOME") >= 150000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 199999.99))
                                         .then(8)
                                         .when((pl.col("MEDIAN_INCOME") >= 200000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 999999.99))
                                         .then(9)
                                         .alias("MEDIAN_INCOME_BRACKET"))
        ses_data = ses_data.rename({"PERSON_ID": "person_id",
                                    "MEDIAN_INCOME": "median_income",
                                    "MEDIAN_INCOME_BRACKET": "median_income_bracket"})

        # compare income and generate
        data = data.join(ses_data, how="inner", on="person_id")
        data = data.with_columns((pl.col("income_bracket") - pl.col("median_income_bracket"))
                                 .alias("compare_to_median_income"))
        # data = data.drop("median_income_bracket")

        return data

    @staticmethod
    def split_string(df, col, split_by, item_index):

        df = df.with_columns((pl.col(col).str.split(split_by).list[item_index]).alias(col))

        return df

    def parse_survey_data(self, smoking=False):  # smoking status will reduce the survey count, hence the option instead
        """
        get survey data of certain questions
        :param smoking: defaults to False; if true, data on smoking frequency is added
        :return: polars dataframe with coded answers
        """
        if smoking:
            self.question_id_dict["smoking_frequency"] = 1585860
        question_ids = tuple(self.question_id_dict.values())

        survey_query = f"SELECT * FROM {self.cdr}.ds_survey WHERE question_concept_id IN {question_ids}"
        survey_data = self.polar_gbq(survey_query)

        # filter out people without survey answer, e.g., skip, don't know, prefer not to answer
        no_answer_ids = survey_data.filter(pl.col("answer").str.contains("PMI"))["person_id"].unique().to_list()
        survey_data = survey_data.filter(~pl.col("person_id").is_in(no_answer_ids))

        # split survey data into separate data by question
        question_list = survey_data["question"].unique().to_list()
        survey_dict = {}
        for question in question_list:
            key_name = question.split(":")[0].split(" ")[0]
            survey_dict[key_name] = survey_data.filter(pl.col("question") == question)
            survey_dict[key_name] = survey_dict[key_name][["person_id", "answer"]]
            survey_dict[key_name] = survey_dict[key_name].rename({"answer": f"{key_name.lower()}_answer"})

        # code income data
        survey_dict["Income"] = survey_dict["Income"].with_columns(pl.col("income_answer").alias("income_bracket"))
        survey_dict["Income"] = survey_dict["Income"].with_columns(pl.col("income_bracket")
                                                                   .replace(self.income_dict, default=pl.first())
                                                                   .cast(pl.Int64))
        survey_dict["Income"] = self.compare_with_median_income(survey_dict["Income"])

        # code education data
        survey_dict["Education"] = survey_dict["Education"].with_columns(
            pl.col("education_answer").alias("education_bracket"))
        survey_dict["Education"] = survey_dict["Education"].with_columns(pl.col("education_bracket")
                                                                         .replace(self.edu_dict, default=pl.first())
                                                                         .cast(pl.Int64))

        # code home own data
        survey_dict["Home"] = self.dummy_coding(data=survey_dict["Home"],
                                                col_name="home_answer",
                                                lookup_dict=self.home_dict)

        # code employment data
        survey_dict["Employment"] = self.dummy_coding(data=survey_dict["Employment"],
                                                      col_name="employment_answer",
                                                      lookup_dict=self.employment_dict)

        # code smoking data
        if smoking:
            survey_dict["Smoking"] = self.dummy_coding(data=survey_dict["Smoking"],
                                                       col_name="smoking_answer",
                                                       lookup_dict=self.smoking_dict)

        # merge data
        data = survey_dict["Income"].join(survey_dict["Education"], how="inner", on="person_id")
        data = data.join(survey_dict["Home"], how="inner", on="person_id")
        data = data.join(survey_dict["Employment"], how="inner", on="person_id")
        if smoking:
            data = data.join(survey_dict["Smoking"], how="left", on="person_id")

        data = self.split_string(df=data, col="income_answer", split_by=": ", item_index=1)
        data = self.split_string(df=data, col="education_answer", split_by=": ", item_index=1)
        data = self.split_string(df=data, col="home_answer", split_by=": ", item_index=1)
        data = self.split_string(df=data, col="employment_answer", split_by=": ", item_index=1)

        data = data.rename(
            {
                "income_answer": "annual income",
                "education_answer": "highest degree",
                "home_answer": "home ownership",
                "employment_answer": "employment status"
            }
        )
        if smoking:
            data = data.rename({"smoking_answer": "smoking status"})

        return data


class Demographic:

    def __init__(
            self,
            ds=os.getenv("WORKSPACE_CDR")
    ):
        self.ds = ds

    def race_ethnicity_query(self):
        query: str = f"""
            SELECT DISTINCT
                p.person_id,
                c1.concept_name AS race,
                c2.concept_name AS ethnicity
            FROM
                {self.ds}.person AS p
            LEFT JOIN
                {self.ds}.concept AS c1 ON p.race_concept_id = c1.concept_id
            LEFT JOIN
                {self.ds}.concept AS c2 ON p.ethnicity_concept_id = c2.concept_id
        """
        return query

    def sex_query(self):
        query: str = f"""
            SELECT
                *
            FROM
                (
                    (
                    SELECT
                        person_id,
                        1 AS sex_at_birth,
                        "male" AS sex
                    FROM
                        {self.ds}.person
                    WHERE
                        sex_at_birth_source_concept_id = 1585846
                    )
                UNION DISTINCT
                    (
                    SELECT
                        person_id,
                        0 AS sex_at_birth,
                        "female" AS sex
                    FROM
                        {self.ds}.person
                    WHERE
                        sex_at_birth_source_concept_id = 1585847
                    )
                )
        """
        return query

    def current_age_query(self):
        query: str = f"""
            SELECT
                DISTINCT p.person_id, 
                EXTRACT(DATE FROM DATETIME(birth_datetime)) AS date_of_birth,
                DATETIME_DIFF(
                    IF(DATETIME(death_datetime) IS NULL, CURRENT_DATETIME(), DATETIME(death_datetime)), 
                    DATETIME(birth_datetime), 
                    DAY
                )/365.2425 AS current_age
            FROM
                {self.ds}.person AS p
            LEFT JOIN
                {self.ds}.death AS d
            ON
                p.person_id = d.person_id
        """
        return query

    def dx_query(self):
        query: str = f"""
            SELECT DISTINCT
                df1.person_id,
                MAX(date) AS last_ehr_date,
                (DATETIME_DIFF(MAX(date), MIN(date), DAY) + 1)/365.2425 AS ehr_length,
                COUNT(code) AS dx_code_occurrence_count,
                COUNT(DISTINCT(code)) AS dx_condition_count,
                DATETIME_DIFF(MAX(date), MIN(birthday), DAY)/365.2425 AS age_at_last_event,
            FROM
                (
                    (
                    SELECT DISTINCT
                        co.person_id,
                        co.condition_start_date AS date,
                        c.concept_code AS code
                    FROM
                        {self.ds}.condition_occurrence AS co
                    INNER JOIN
                        {self.ds}.concept AS c
                    ON
                        co.condition_source_value = c.concept_code
                    WHERE
                        c.vocabulary_id IN ("ICD9CM", "ICD10CM")
                    )
                UNION DISTINCT
                    (
                    SELECT DISTINCT
                        co.person_id,
                        co.condition_start_date AS date,
                        c.concept_code AS code
                    FROM
                        {self.ds}.condition_occurrence AS co
                    INNER JOIN
                        {self.ds}.concept AS c
                    ON
                        co.condition_source_concept_id = c.concept_id
                    WHERE
                        c.vocabulary_id IN ("ICD9CM", "ICD10CM")
                    )
                UNION DISTINCT
                    (
                    SELECT DISTINCT
                        o.person_id,
                        o.observation_date AS date,
                        c.concept_code AS code
                    FROM
                        {self.ds}.observation AS o
                    INNER JOIN
                        {self.ds}.concept AS c
                    ON
                        o.observation_source_value = c.concept_code
                    WHERE
                        c.vocabulary_id IN ("ICD9CM", "ICD10CM")
                    )
                UNION DISTINCT
                    (
                    SELECT DISTINCT
                        o.person_id,
                        o.observation_date AS date,
                        c.concept_code AS code
                    FROM
                        {self.ds}.observation AS o
                    INNER JOIN
                        {self.ds}.concept AS c
                    ON
                        o.observation_source_concept_id = c.concept_id
                    WHERE
                        c.vocabulary_id IN ("ICD9CM", "ICD10CM")
                    )
                ) AS df1
            INNER JOIN
                (
                    SELECT
                        person_id, 
                        EXTRACT(DATE FROM DATETIME(birth_datetime)) AS birthday
                    FROM
                        {self.ds}.person
                ) AS df2
            ON
                df1.person_id = df2.person_id
            GROUP BY 
                df1.person_id
        """
        return query

    def get_demographic_data(
            self,
            cohort_csv_file_path,
            output_csv_file_path=None,
            current_age=False,
            sex=False,
            race_ethnicity=False,
            diagnosis=False
    ):
        # Load data
        cohort_df = pl.read_csv(cohort_csv_file_path)

        print("Getting demographic data...")
        demo_df = cohort_df
        if current_age:
            current_age_df = PT.polars_gbq(self.current_age_query())
            demo_df = demo_df.join(current_age_df, how="left", on="person_id")
        if sex:
            sex_df = PT.polars_gbq(self.sex_query())
            demo_df = demo_df.join(sex_df, how="left", on="person_id")
        if race_ethnicity:
            race_ethnicity_df = PT.polars_gbq(self.race_ethnicity_query())
            demo_df = demo_df.join(race_ethnicity_df, how="left", on="person_id")
        if diagnosis:
            dx_df = PT.polars_gbq(self.dx_query())
            demo_df = demo_df.join(dx_df, how="left", on="person_id")
        if output_csv_file_path is None:
            output_csv_file_path = "cohort_with_demographic_data.csv"
        demo_df.write_csv(output_csv_file_path)
        print("Done.")
        print()
        print(f"Demographic data saved to {output_csv_file_path}")

    @staticmethod
    def create_table_one(
            cohort_csv_file_path,
            columns_to_use: list,
            group_by: str,
            missing=False,
            include_null=True
    ):
        # load cohort data
        df = pl.read_csv(cohort_csv_file_path)

        # create table one
        table_one = TableOne(
            data=df[columns_to_use].to_pandas(),
            groupby=group_by,
            missing=missing,
            include_null=include_null
        )

        return table_one


class GWAS:

    def __init__(self):
        pass

    @staticmethod
    def generate_sh_script(script_name, commands):
        with open(script_name, 'w') as f:
            f.write("#!/bin/bash\n")  # Shebang line for bash
            for command in commands:
                f.write(command + "\n")

        # Make script executable
        import os
        os.chmod(script_name, 0o755)

        print(f"Generated script: {script_name}")


    @staticmethod
    def generate_plink2_variant_filter_script(
            script_name: str = "variant_filter.sh",
            hwe_threshold: float = 0.000001,
            geno_threshold: float = 0.1,
            mind_threshold: float = 0.1,
            maf_threshold: float = 0.01,
            biallelic_only: bool = True,
            split_multi_allelic: bool = False,
            custom_args: str = None,
    ):
        """
        Generate a simple PLINK2 filtering script.

        :param custom_args: Extra args to be used
        :param script_name: Name of the output shell script
        :param hwe_threshold: Hardy-Weinberg's equilibrium p-value threshold
        :param geno_threshold: Genotype missingness threshold
        :param mind_threshold: Individual missingness threshold
        :param maf_threshold: Minor allele frequency threshold
        :param split_multi_allelic: split multi-allelic alleles or not
        :param biallelic_only: use only biallelic alleles or not
        """

        prerun_command = "PLINK_OUTPUT_BASE=$(echo $OUTPUT_PGEN | sed 's/.pgen$//g')"

        plink_command = "plink2 --pgen $INPUT_PGEN --pvar $INPUT_PVAR --psam $INPUT_PSAM --make-pgen --no-fid --out $PLINK_OUTPUT_BASE"
        if hwe_threshold:
            plink_command += f" --hwe {hwe_threshold}"
        if geno_threshold:
            plink_command += f" --geno {geno_threshold}"
        if mind_threshold:
            plink_command += f" --mind {mind_threshold}"
        if maf_threshold:
            plink_command += f" --maf {maf_threshold}"
        if biallelic_only:
            plink_command += f" --max-alleles 2"
        if split_multi_allelic:
            plink_command += f" --split_multiallelic"
        if custom_args is not None:
            plink_command += f" {custom_args}"

        # add FID for psam
        postrun_command = "echo -e '#FID\\tIID\\tSEX' > ${PLINK_OUTPUT_BASE}.tmp; cat \"${PLINK_OUTPUT_BASE}.psam\" | tail -n +2 | awk -F '\\t' -v 'OFS=\\t' '{ print $1, $1, $2 }' >> ${PLINK_OUTPUT_BASE}.tmp; mv ${PLINK_OUTPUT_BASE}.tmp ${PLINK_OUTPUT_BASE}.psam"
        postrun_command += "\necho 'Added FID to psam file'"
        postrun_command += "\nhead -n 5 ${PLINK_OUTPUT_BASE}.psam"

        script_commands = [
            prerun_command,
            plink_command,
            postrun_command
        ]

        GWAS.generate_sh_script(script_name=script_name, commands=script_commands)

    @staticmethod
    def generate_regenie_gwas_script(
            script_name: str = "regenie_gwas.sh",
            pgen_prefix: str = None,
            output_prefix: str = None,
            threads: int = 4,
            step1_block_size: int = 1000,
            step2_block_size: int = 400,
            step1_custom_args: str = None,
            step2_custom_args: str = None,
    ):
        """
        Generate a simple REGENIE GWAS script compatible with REGENIE v4.1.
        """

        if pgen_prefix is None:
            pgen_prefix = "$PLINK_OUTPUT_BASE"

        if output_prefix is None:
            output_prefix = "REGENIE_OUTPUT_BASE"

        prerun_command = "REGENIE_OUTPUT_BASE=$(echo $REGENIE_OUTPUT_FILES | sed 's/\*$//')"

        base_script = f"regenie --pgen {pgen_prefix} --phenoFile $INPUT_PHENO --covarFile $INPUT_COV --threads {threads}"
        step1_script = base_script + " --step 1"
        step2_script = base_script + " --step 2"

        # step 1
        step1_script += f" --out ${{{output_prefix}}}_gwas_step1"
        step1_script += f" --bsize {step1_block_size}"
        if step1_custom_args is not None:
            step1_script += f" {step1_custom_args}"

        # step 2
        step2_script += f" --out ${{{output_prefix}}}_gwas_step2"
        step2_script += f" --pred ${{{output_prefix}}}_gwas_step1_pred.list"
        step2_script += f" --bsize {step2_block_size}"
        step2_script += f" --firth --approx"
        if step2_custom_args is not None:
            step2_script += f" {step2_custom_args}"

        script_commands = [prerun_command, step1_script, step2_script]

        GWAS.generate_sh_script(script_name=script_name, commands=script_commands)

    @staticmethod
    def prepare_regenie_inputs(
            complete_table_path: str,
            pheno_cols: list,
            cov_cols: list,
            iid_col: str,
            fid_col: str = None,
            input_seperator: str = ",",
            schema_dict=None,
            output_prefix: str = ""
    ):
        """
        Generate phenotype.txt and covariate.txt to run GWAS with regenie.
        NOTE: this function will calculate the average phenotype (score) for each person since each has multiple values.
        This would only work for a single continuous phenotype and will need to generalize for multiple phenotypes.
        """
        if schema_dict is None:
            schema_dict = {}

        complete_table = pl.read_csv(f"{complete_table_path}", separator=input_seperator, schema_overrides=schema_dict)
        cols = [iid_col] + pheno_cols + cov_cols
        if fid_col is not None:
            cols += [fid_col]
        complete_table = complete_table.unique()

        # prepare FID & IID
        if fid_col is None:
            complete_table = complete_table.with_columns(pl.col(iid_col).alias("FID"))
        else:
            if fid_col != "FID":
                complete_table = complete_table.rename({fid_col: "FID"})
        if iid_col != "IID":
            complete_table = complete_table.rename({iid_col: "IID"})

        # phenotypes
        pheno_table = complete_table[["FID", "IID"] + pheno_cols].unique()
        pheno_table = pheno_table.group_by(["FID", "IID"]).agg(
            pl.col("lnk").mean().alias("lnk"))  # need to generalize for multiple phenotypes
        pheno_file = f"{output_prefix}phenotypes.txt"
        print(f"Phenotype file saved as {pheno_file}")
        pheno_table.write_csv(pheno_file, separator="\t")
        print()

        # covariates
        cov_table = complete_table[["FID", "IID"] + cov_cols].unique()
        cov_file = f"{output_prefix}covariates.txt"
        print(f"Covariate file saved as {cov_file}")
        cov_table.write_csv(cov_file, separator="\t")
        print()

        # unique combined table
        unique_combined_table = pheno_table.join(cov_table, how="inner", on=["FID", "IID"])
        unique_combined_table = unique_combined_table.with_columns(pl.col("IID").alias("person_id"))
        name, ext = os.path.splitext(complete_table_path)
        combined_file = name + ".txt"
        print(f"Combined file saved as {combined_file}")
        unique_combined_table.write_csv(combined_file, separator=",")
        print()

    @staticmethod
    def merge_scripts(
            script_file_list: list,
            output_file_name: str
    ):
        """
        Merge shell scripts, removing the first line (shebang) from all but the first script.

        Parameters:
        script_files: list of script file paths
        output_file: output file path
        """
        with open(output_file_name, 'w') as outfile:
            for i, script_file in enumerate(script_file_list):
                with open(script_file, 'r') as infile:
                    lines = infile.readlines()

                # Skip the first line for subsequent scripts (remove shebang)
                start_line = 1 if i > 0 else 0

                for line in lines[start_line:]:
                    outfile.write(line)

                # Add a newline between scripts
                if i < len(script_file_list) - 1:
                    outfile.write('\n')

    @staticmethod
    def run_gwas_dsub(
        regenie_input_pheno_file_path: str,
        regenie_input_cov_file_path: str,
        regenie_threads: int = 4,
        regenie_output_folder: str = None,
        regenie_step1_custom_args: str = None,
        regenie_step2_custom_args: str = None,
        plink_hwe_threshold: float = 0.000001,
        plink_geno_threshold: float = 0.1,
        plink_mind_threshold: float = 0.1,
        plink_maf_threshold: float = 0.01,
        plink_biallelic_only: bool = True,
        plink_split_multi_allelic: bool = False,
        plink_input_folder: str = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/pgen/",
        plink_input_file_prefix: str = "acaf_threshold.chr",
        plink_output_folder: str = None,
        plink_custom_args: str = None,
        dsub_job_prefix: str = f"dsub_{datetime.datetime.now().strftime('%Y%m%d')}",
        dsub_env_dict=None,
        dsub_machine_type: str = "c4d-highcpu-8",
        dsub_disk_type: str = "hyperdisk-balanced",
        dsub_region: str = "us-central1",
        dsub_docker_image: str = "gcr.io/ni-nhgri-phis-comp-initiative/gptk:0.1",
        dsub_custom_args: str = None,
        dsub_preemptible: bool = False,
        dsub_show_command: bool = False,
        chr_list=None,  # exclude sex chromosome
    ):
        regenie_output_folder = regenie_output_folder.rstrip("/")
        plink_input_folder = plink_input_folder.rstrip("/")
        plink_output_folder = plink_output_folder.rstrip("/")

        if dsub_env_dict is None:
            dsub_env_dict = {}
        if chr_list is None:
            chr_list = list(range(1, 23))
        dsub_job_prefix = dsub_job_prefix.replace("_", "-")

        # Generate PLINK2 script to filter variant from pgen
        print("Generating PLINK2 script to filter variant...")
        plink_script_name = "variant_filter.sh"
        GWAS.generate_plink2_variant_filter_script(
            script_name=plink_script_name,
            hwe_threshold=plink_hwe_threshold,
            geno_threshold=plink_geno_threshold,
            mind_threshold=plink_mind_threshold,
            maf_threshold=plink_maf_threshold,
            biallelic_only=plink_biallelic_only,
            split_multi_allelic=plink_split_multi_allelic,
            custom_args=plink_custom_args,
        )
        print()

        # Generate REGENIE script to run gwas
        print("Generating REGENIE script to run GWAS...")
        regenie_script_name = "regenie_gwas.sh"
        GWAS.generate_regenie_gwas_script(
            script_name=regenie_script_name,
            threads=regenie_threads,
            step1_custom_args=regenie_step1_custom_args,
            step2_custom_args=regenie_step2_custom_args,
        )
        print()

        # Merge scripts
        print("Merging PLINK2 and REGENIE scripts...")
        merged_script_name = "plink_regenie_gwas.sh"
        GWAS.merge_scripts(
            script_file_list=[plink_script_name, regenie_script_name],
            output_file_name=merged_script_name
        )
        print()

        # Run GWAS with dsub
        dsub_jobs = {}
        for i in chr_list:
            job_name = f"{dsub_job_prefix}-chr{i}"

            plink_input_base = f"{plink_input_folder}/{plink_input_file_prefix}{i}"
            plink_output_base = f"{plink_output_folder}/{dsub_job_prefix}__filtered_{plink_input_file_prefix}{i}"

            regenie_output_base = f"{regenie_output_folder}/{dsub_job_prefix}__chr{i}"
            regenie_input_pheno = f"{regenie_input_pheno_file_path}"
            regenie_input_cov = f"{regenie_input_cov_file_path}"

            dsub_job = Dsub(
                machine_type=dsub_machine_type,
                disk_type=dsub_disk_type,
                docker_image=dsub_docker_image,
                job_script_name=merged_script_name,
                job_name=job_name,
                input_dict={
                    "INPUT_PGEN": f"{plink_input_base}.pgen",
                    "INPUT_PVAR": f"{plink_input_base}.pvar",
                    "INPUT_PSAM": f"{plink_input_base}.psam",
                    "INPUT_PHENO": f"{regenie_input_pheno}",
                    "INPUT_COV": f"{regenie_input_cov}"
                },
                output_dict={
                    "OUTPUT_PGEN": f"{plink_output_base}.pgen",
                    "OUTPUT_PVAR": f"{plink_output_base}.pvar",
                    "OUTPUT_PSAM": f"{plink_output_base}.psam",
                    "REGENIE_OUTPUT_FILES": f"{regenie_output_base}*"
                },
                env_dict=dsub_env_dict,
                region=dsub_region,
                custom_args=dsub_custom_args,
                preemptible=dsub_preemptible,
            )
            dsub_jobs[job_name] = dsub_job
            dsub_job.run(show_command=dsub_show_command)

        # Save to file
        with open(f"{dsub_job_prefix}.pkl", "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(dsub_jobs, f)

        print("To check all gwas jobs, use method .check_status(dsub_jobs, show_all=True).\n"
              "For example, if class GWAS was instantiated as gwas = GWAS() and dsub run as dsub_jobs=gwas.run_gwas_dsub,"
              "the command would be gwas.check_status(dsub_jobs, show_all=True)")
        print()
        print("To check individual job status, use gwas.check_gwas_jobs(dsub_jobs, show_all=False, job_name={your_job_name})")
        print()
        print("Similarly, to kill all jobs use gwas.kill(dsub_jobs, kill_all=True),"
              "or gwas.kill(dsub_jobs, kill_all=False, job_name={your_job_name}) to kill an individual job.")
        print()
        print(f"dsub_jobs dict was saved as {dsub_job_prefix}.pkl. To load, use method .load_pickle(<pickle-file>)")
        print()

        return dsub_jobs

    @staticmethod
    def load_pickle(file):
        with open(file, "rb") as f:
            pickle_obj = pickle.load(f)
        return pickle_obj

    @staticmethod
    def check_status(
            dsub_jobs: dict = None,
            show_all: bool = True,
            job_name: str = None,
            full: bool = True,
            streaming: bool = True,
            update_interval: int = 10,
            job_limit: int = None
    ):
        if job_limit is None:
            job_limit = len(dsub_jobs)
        if show_all:
            dsub_user = os.getenv("OWNER_EMAIL").split("@")[0]
            command = f"dstat --project $GOOGLE_PROJECT --users {dsub_user} --status '*' --limit {job_limit}"
            if streaming:
                # Auto-detect notebook
                try:
                    # noinspection PyUnresolvedReferences
                    from IPython.display import clear_output
                    is_notebook = True
                except ImportError:
                    is_notebook = False

                while True:
                    # Clear output
                    if is_notebook:
                        clear_output(wait=True)
                    else:
                        os.system('clear' if os.name == 'posix' else 'cls')

                    # Run command and print output
                    subprocess.run([command], shell=True)

                    # Wait
                    time.sleep(update_interval)

            else:
                subprocess.run([command], shell=True)

        else:
            if job_name is not None:
                assert isinstance(dsub_jobs[job_name], Dsub)
                print(job_name)
                dsub_jobs[job_name].check_status(full=full)
                print()
            else:
                print("Please provide individual job name to show status. To show all, use show_all=True")

    @staticmethod
    def kill(dsub_jobs: dict = None, kill_all: bool = False, job_name: str = None):
        if kill_all:
            for k, v in dsub_jobs.items():
                assert isinstance(v, Dsub)
                print(k)
                v.kill()
                print()
        else:
            if job_name is not None:
                assert isinstance(dsub_jobs[job_name], Dsub)
                print(job_name)
                dsub_jobs[job_name].kill()
                print()
            else:
                print("Please provide individual job name to kill. To kill all, use kill_all=True")
