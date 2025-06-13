from google.cloud import bigquery
from tableone import TableOne

import datetime
import os
import polars as pl
import subprocess
import sys
import tctk.PolarsTools as PT
import yaml


class Dsub:
    """
    This class is a wrapper to run dsub on the All of Us researcher workbench.
    input_dict and output_dict values must be paths to Google Cloud Storage bucket(s).
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

    def _dsub_script(self):

        base_script = (
            f"dsub" + " " +
            f"--provider \"{self.provider}\"" + " " +
            f"--regions \"{self.region}\"" + " " +
            f"--machine-type \"{self.machine_type}\"" + " " +
            f"--disk-type \"{self.disk_type}\"" + " " +
            f"--boot-disk-size {self.boot_disk_size}" + " " +
            f"--disk-size {self.disk_size}" + " " +
            f"--user-project \"{self.user_project}\"" + " " +
            f"--project \"{self.project}\"" + " " +
            f"--image \"{self.docker_image}\"" + " " +
            f"--network \"network\"" + " " +
            f"--subnetwork \"subnetwork\"" + " " +
            f"--service-account \"$(gcloud config get-value account)\"" + " " +
            f"--user \"{self.dsub_user_name}\"" + " " +
            f"--logging {self.log_file_path} $@" + " " +
            f"--name \"{self.job_name}\"" + " " +
            f"--env GOOGLE_PROJECT=\"{self.google_project}\"" + " "
        )

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
        script = base_script + env_flags + input_flags + output_flags + job_script

        # add preemptible argument if used
        if self.preemptible:
            script += " --preemptible"

        # add attribute for convenience
        self.script = script

        return script

    def check_status(self, streaming=False):

        # base command
        check_status = (
            f"dstat --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\" --status \"*\""
        )

        # streaming status
        if streaming:
            check_status += " --wait --poll-interval 60"
            process = subprocess.Popen(
                [check_status],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            try:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip(), flush=True)
            except KeyboardInterrupt:
                process.kill()
                sys.exit(0)
        # full static status
        else:
            check_status += " --full"
            subprocess.run([check_status], shell=True)

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

        kill_job = (
            f"ddel --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\""
        )
        subprocess.run([kill_job], shell=True)

    def run(self, show_command=False):

        s = subprocess.run([self._dsub_script()], shell=True, capture_output=True, text=True)

        if s.returncode == 0:
            print(f"Successfully run dsub to schedule job {self.job_name}.")
            self.job_id = s.stdout.strip()
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


class DsubUtils:

    """Utilities for dsub"""

    @staticmethod
    def extract_events(dsub_status_text, latest_only=False):
        """
        Extract events section from dsub status output.

        :param dsub_status_text: string containing the full dsub status output
        :type dsub_status_text: str
        :param latest_only: if True, return only the most recent event with job status
        :type latest_only: bool
        :return: list of event dictionaries with name and start-time, or single dict if latest_only=True
        :rtype: list[dict] or dict
        """
        try:
            # Parse the YAML content
            data = yaml.safe_load(dsub_status_text)

            # Extract events
            events = data.get('events', [])

            if latest_only:
                # Return latest event with job status
                if not events:
                    return {
                        'latest_event': None,
                        'event_time': None,
                        'job_status': data.get('status', 'UNKNOWN'),
                        'status_message': data.get('status-message', '')
                    }

                # Events are typically in chronological order, so take the last one
                latest_event = events[-1]

                return {
                    'latest_event': latest_event.get('name', ''),
                    'event_time': latest_event.get('start-time', ''),
                    'job_status': data.get('status', 'UNKNOWN'),
                    'status_message': data.get('status-message', '')
                }

            # Return all events
            formatted_events = []
            for event in events:
                formatted_event = {
                    'name': event.get('name', ''),
                    'start_time': event.get('start-time', '')
                }
                formatted_events.append(formatted_event)

            return formatted_events

        except Exception as e:
            print(f"Error parsing events: {e}")
            if latest_only:
                return {
                    'latest_event': None,
                    'event_time': None,
                    'job_status': 'ERROR',
                    'status_message': f'Parse error: {e}'
                }
            return []


    @staticmethod
    def extract_job_summary(dsub_status_text):
        """
        Extract job summary section from script-name to the end.

        :param dsub_status_text: string containing the full dsub status output
        :type dsub_status_text: str
        :return: dictionary containing script-name through status fields
        :rtype: dict
        """
        try:
            # Parse the YAML content
            data = yaml.safe_load(dsub_status_text)

            # Extract the summary fields (from script-name onwards)
            summary_fields = [
                'script-name', 'start-time', 'status',
                'status-detail', 'status-message'
            ]

            summary = {}
            for field in summary_fields:
                if field in data:
                    summary[field] = data[field]

            # Also include script content if present
            if 'script' in data:
                summary['script'] = data['script']

            return summary

        except Exception as e:
            print(f"Error parsing job summary: {e}")
            return {}


    @staticmethod
    def parse_dsub_status(dsub_status_text):
        """
        Parse complete dsub status and return both events and summary.

        :param dsub_status_text: string containing the full dsub status output
        :type dsub_status_text: str
        :return: tuple containing (events_list, summary_dict)
        :rtype: tuple[list[dict], dict]
        """
        events = DsubUtils.extract_events(dsub_status_text)
        summary = DsubUtils.extract_job_summary(dsub_status_text)

        return events, summary


    @staticmethod
    def print_events_summary(events):
        """
        Helper function to print events in a readable format.

        :param events: list of event dictionaries
        :type events: list[dict]
        :return: None
        :rtype: None
        """
        print("Job Events Timeline:")
        print("-" * 50)
        for event in events:
            print(f"{event['name']:20} | {event['start_time']}")


    @staticmethod
    def print_job_summary(summary):
        """
        Helper function to print job summary in a readable format.

        :param summary: dictionary containing job summary information
        :type summary: dict
        :return: None
        :rtype: None
        """
        print("\nJob Summary:")
        print("-" * 50)
        for key, value in summary.items():
            if key == 'script':
                print(f"{key:15} | [Script content - {len(value.splitlines())} lines]")
            else:
                print(f"{key:15} | {value}")


    @staticmethod
    def aggregate_dsub_status(status_list):
        """
        Aggregate multiple dsub status outputs into a summary table.

        :param status_list: list of dsub status text strings or list of dictionaries
        :type status_list: list[str] or list[dict]
        :return: list of dictionaries with job summary information
        :rtype: list[dict]
        """
        summary_table = []

        for i, status_item in enumerate(status_list):
            try:
                # Handle if input is already parsed dict or raw text
                if isinstance(status_item, dict):
                    data = status_item
                else:
                    data = yaml.safe_load(status_item)

                # Get latest event info
                events = data.get('events', [])
                if events:
                    latest_event = events[-1].get('name', 'unknown')
                    latest_event_time = events[-1].get('start-time', '')
                else:
                    latest_event = 'no-events'
                    latest_event_time = ''

                # Build summary row
                job_summary = {
                    'job_name': data.get('job-name', f'job-{i}'),
                    'job_id': data.get('job-id', ''),
                    'last_event': latest_event,
                    'last_event_time': latest_event_time,
                    'status': data.get('status', 'UNKNOWN'),
                    'status_message': data.get('status-message', ''),
                    'start_time': data.get('start-time', ''),
                    'end_time': data.get('end-time', ''),
                    'duration_minutes': DsubUtils.calculate_duration(data.get('start-time', ''), data.get('end-time', ''))
                }

                summary_table.append(job_summary)

            except Exception as e:
                # Handle parsing errors gracefully
                error_summary = {
                    'job_name': f'parse-error-{i}',
                    'job_id': '',
                    'last_event': 'error',
                    'last_event_time': '',
                    'status': 'PARSE_ERROR',
                    'status_message': str(e),
                    'start_time': '',
                    'end_time': '',
                    'duration_minutes': 0
                }
                summary_table.append(error_summary)

        return summary_table


    @staticmethod
    def calculate_duration(start_time, end_time):
        """
        Calculate duration between start and end times in minutes.

        :param start_time: start time string
        :type start_time: str
        :param end_time: end time string
        :type end_time: str
        :return: duration in minutes
        :rtype: float
        """
        try:
            if not start_time or not end_time:
                return 0.0

            # Parse times (handle different formats)
            from dateutil import parser
            start = parser.parse(start_time)
            end = parser.parse(end_time)

            duration = (end - start).total_seconds() / 60.0
            return round(duration, 1)

        except Exception:
            return 0.0


    @staticmethod
    def print_status_table(summary_table, show_all_columns=False):
        """
        Print the aggregated status table in a readable format.

        :param summary_table: list of job summary dictionaries
        :type summary_table: list[dict]
        :param show_all_columns: whether to show all columns or just key ones
        :type show_all_columns: bool
        :return: None
        :rtype: None
        """
        if not summary_table:
            print("No jobs to display")
            return

        # Define column widths
        if show_all_columns:
            headers = ['Job Name', 'Job ID', 'Last Event', 'Status', 'Duration (min)', 'Status Message']
            widths = [20, 25, 15, 10, 12, 30]
        else:
            headers = ['Job Name', 'Last Event', 'Status', 'Duration (min)']
            widths = [25, 20, 12, 12]

        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for job in summary_table:
            if show_all_columns:
                row_data = [
                    job['job_name'][:19],
                    job['job_id'][:24],
                    job['last_event'][:14],
                    job['status'][:9],
                    str(job['duration_minutes']),
                    job['status_message'][:29]
                ]
            else:
                row_data = [
                    job['job_name'][:24],
                    job['last_event'][:19],
                    job['status'][:11],
                    str(job['duration_minutes'])
                ]

            row_line = " | ".join(data.ljust(w) for data, w in zip(row_data, widths))
            print(row_line)


    @staticmethod
    def save_status_table_csv(summary_table, filename="dsub_status_summary.csv"):
        """
        Save the status table to a CSV file.

        :param summary_table: list of job summary dictionaries
        :type summary_table: list[dict]
        :param filename: output CSV filename
        :type filename: str
        :return: None
        :rtype: None
        """
        import csv

        if not summary_table:
            print("No data to save")
            return

        # Get all possible keys from all dictionaries
        all_keys = set()
        for job in summary_table:
            all_keys.update(job.keys())

        fieldnames = sorted(all_keys)

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_table)

        print(f"Status table saved to {filename}")


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
