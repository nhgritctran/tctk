from tableone import TableOne

import os
import pandas as pd
import polars as pl
import tctk.polars_tools as pt


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
            current_age_df = pt.polars_gbq(self.current_age_query())
            demo_df = demo_df.join(current_age_df, how="left", on="person_id")
        if sex:
            sex_df = pt.polars_gbq(self.sex_query())
            demo_df = demo_df.join(sex_df, how="left", on="person_id")
        if race_ethnicity:
            race_ethnicity_df = pt.polars_gbq(self.race_ethnicity_query())
            demo_df = demo_df.join(race_ethnicity_df, how="left", on="person_id")
        if diagnosis:
            dx_df = pt.polars_gbq(self.dx_query())
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
            include_null=True,
            category_orders: dict = None,  # e.g. {"age": age_order, ...}
            **kwargs
    ):
        # load cohort data
        df = pl.read_csv(cohort_csv_file_path).to_pandas()[columns_to_use]

        # Force categorical ordering directly on the DataFrame
        if category_orders:
            for col, order in category_orders.items():
                if col in df.columns:
                    df[col] = pd.Categorical(df[col], categories=order, ordered=True)

        # create table one
        table_one = TableOne(
            data=df,
            groupby=group_by,
            missing=missing,
            include_null=include_null
        )

        return table_one