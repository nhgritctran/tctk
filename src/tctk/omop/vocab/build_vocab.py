"""
Convert Athena OMOP vocabulary CSV files to a DuckDB database.

Usage:
    # Run from the folder containing Athena CSVs:
    python build_vocab.py

    # Or specify paths:
    python build_vocab.py --csv-dir /path/to/csvs --output vocab.duckdb

    # Overwrite existing database:
    python build_vocab.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import duckdb

REQUIRED_TABLES = {
    "concept": "CONCEPT.csv",
    "concept_synonym": "CONCEPT_SYNONYM.csv",
    "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
    "concept_ancestor": "CONCEPT_ANCESTOR.csv",
}


def build_vocab_db(csv_dir: str, output_path: str, overwrite: bool = False) -> str:
    csv_dir = Path(csv_dir)
    db_path = Path(output_path)

    if not csv_dir.is_dir():
        print(f"Error: CSV directory not found: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    missing = [f for t, f in REQUIRED_TABLES.items() if not (csv_dir / f).is_file()]
    if missing:
        print(f"Error: Missing CSV files in {csv_dir}: {', '.join(missing)}", file=sys.stderr)
        print("Download from https://athena.ohdsi.org", file=sys.stderr)
        sys.exit(1)

    if db_path.exists():
        if overwrite:
            db_path.unlink()
            print(f"Removed existing: {db_path}")
        else:
            print(f"Error: {db_path} already exists. Use --overwrite to rebuild.", file=sys.stderr)
            sys.exit(1)

    print(f"Building: {db_path}")
    print(f"Source:   {csv_dir}")

    conn = duckdb.connect(str(db_path))

    try:
        for table_name, filename in REQUIRED_TABLES.items():
            filepath = csv_dir / filename
            print(f"  Loading {filename}...", end=" ", flush=True)

            conn.execute(f"""
                CREATE TABLE {table_name} AS
                SELECT *
                FROM read_csv_auto(
                    '{filepath}',
                    delim='\t',
                    header=true,
                    quote=''
                )
            """)

            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"{row_count:,} rows")

        print("  Creating indexes...", end=" ", flush=True)

        conn.execute("CREATE INDEX idx_concept_id ON concept(concept_id)")
        conn.execute("CREATE INDEX idx_concept_domain ON concept(domain_id)")
        conn.execute("CREATE INDEX idx_concept_vocab ON concept(vocabulary_id)")
        conn.execute("CREATE INDEX idx_concept_standard ON concept(standard_concept)")

        conn.execute("CREATE INDEX idx_cs_concept_id ON concept_synonym(concept_id)")

        conn.execute("CREATE INDEX idx_cr_id1 ON concept_relationship(concept_id_1)")
        conn.execute("CREATE INDEX idx_cr_id2 ON concept_relationship(concept_id_2)")
        conn.execute("CREATE INDEX idx_cr_rel ON concept_relationship(relationship_id)")

        conn.execute("CREATE INDEX idx_ca_desc ON concept_ancestor(descendant_concept_id)")
        conn.execute("CREATE INDEX idx_ca_anc ON concept_ancestor(ancestor_concept_id)")

        print("done")

        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"\nDone: {db_path} ({size_mb:.1f} MB)")

    finally:
        conn.close()

    return str(db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Athena OMOP vocabulary CSVs to DuckDB."
    )
    parser.add_argument(
        "--csv-dir",
        default=".",
        help="Directory containing Athena CSV files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="vocab.duckdb",
        help="Output DuckDB file path (default: vocab.duckdb)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing database if it exists",
    )

    args = parser.parse_args()
    build_vocab_db(args.csv_dir, args.output, args.overwrite)
