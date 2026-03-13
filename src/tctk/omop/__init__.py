from huggingface_hub import hf_hub_download


def get_vocab_db():
    """Download (if needed) and return the path to the OMOP vocab DuckDB file."""
    return hf_hub_download(
        repo_id="tctran/tctk-omop-vocab",
        filename="vocab.duckdb",
        repo_type="dataset",
    )
