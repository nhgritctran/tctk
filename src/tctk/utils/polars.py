from google.cloud import bigquery
import polars as pl
import pyarrow as pa


def spark_to_polars(spark_df):
    """
    Convert spark df to polars df
    :param spark_df: spark df
    :return: polars df
    """
    # noinspection PyProtectedMember,PyArgumentList
    polars_df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))

    return polars_df


def polars_gbq(query, remove_metadata=False):
    """
    Take a SQL query and return result as polars dataframe
    :param query: BigQuery SQL query
    :param remove_metadata: remove metadata from bigquery dataframe, mainly to avoid Google custom datetime type issue
    :return: polars dataframe
    """
    client = bigquery.Client()
    query_job = client.query(query)
    rows = query_job.result()
    table = rows.to_arrow()
    # this step is a walk-around for BigQuery custom date-time types
    if remove_metadata:
        table = pa.Table.from_batches(
            table.to_batches(),
            schema=pa.schema([field.remove_metadata() for field in table.schema])
        )
    df = pl.from_arrow(table)

    return df
