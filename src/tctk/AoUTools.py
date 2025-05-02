from google.cloud import bigquery
from tableone import TableOne

import datetime
import os
import polars as pl
import subprocess
import sys
import tctk.PolarsTools as PT


class dsub:
    """
    This class is a wrapper to run dsub on the All of Us researcher workbench.
    """

    def __init__(
        self,
        docker_image: str,
        job_script_name: str,
        job_name: str,
        input_files_dict: {},
        multiple_output_files=False,
        output_file_name="",
        output_file_pattern="",
        output_folder=None,
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
        self.input_files_dict = input_files_dict
        self.multiple_output_files = multiple_output_files
        self.output_file_name = output_file_name
        self.output_file_pattern = output_file_pattern
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

        # output folder
        if output_folder is not None:
            self.output_folder = output_folder
        else:
            self.output_folder = (
                f"{self.bucket}/dsub/results/{self.job_name}/{self.user_name}/{self.date}/{self.time}"
            )
        self.phewas_output_file = (
            f"/mnt/data/output/{self.output_folder.replace(':/', '')}/{self.output_file_name}"
        )

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
        if len(self.input_files_dict) > 0:
            for k, v in self.input_files_dict.items():
                input_flags += f"--input {k}={v}" + " "

        # generate output flag
        output_flag = ""
        if self.output_file_name != "":
            if self.multiple_output_files:
                if self.output_file_pattern != "":
                    output_flag += f"--output OUTPUT_FILES={self.output_folder}/{self.output_file_pattern}" + " "
                else:
                    print("Multiple output files require output_file_pattern.")
                    sys.exit(1)
            else:
                output_flag += f"--output OUTPUT_FILE={self.output_folder}/{self.output_file_name}" + " "
            output_flag += f"--env PHEWAS_OUTPUT_FILE={self.phewas_output_file}" + " "
        else:
            print("output_file_name is required.")
            sys.exit(1)

        # job script flag
        job_script = f"--script {self.job_script_name}"

        # combined script
        script = base_script + input_flags + output_flag + job_script

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
                        1 AS sex_at_birth
                    FROM
                        {self.ds}.person
                    WHERE
                        sex_at_birth_source_concept_id = 1585846
                    )
                UNION DISTINCT
                    (
                    SELECT
                        person_id,
                        0 AS sex_at_birth
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
            output_csv_file_path="cohort_with_demographic_data.csv",
            current_age=True,
            sex=True,
            race_ethnicity=True,
            diagnosis=True
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
        demo_df.write_csv(output_csv_file_path)
        print("Done.")
        print()
        print(f"Demographic data saved to {output_csv_file_path}")

    @staticmethod
    def create_table_one(cohort_csv_file_path, columns_to_use: list, group_by: str):
        # load cohort data
        df = pl.read_csv(cohort_csv_file_path)

        # create table one
        table_one = TableOne(df[columns_to_use].to_pandas(), groupby=group_by)

        return table_one
