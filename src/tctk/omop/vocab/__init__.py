from huggingface_hub import hf_hub_download


def get_vocab_db(force_download=False):
    """Download (if needed) and return the path to the OMOP vocab DuckDB file.

    Args:
        force_download (bool): If True, re-download even if cached locally. Default False.
    """
    return hf_hub_download(
        repo_id="tctran/tctk-omop-vocab",
        filename="vocab.duckdb",
        repo_type="dataset",
        force_download=force_download,
    )
