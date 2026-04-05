import datetime
import os
import pickle
import polars as pl
import subprocess
import time

from tctk.aou.dsub import Dsub


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

        Args:
            script_name (str): Name of the output shell script
            hwe_threshold (float): Hardy-Weinberg's equilibrium p-value threshold
            geno_threshold (float): Genotype missingness threshold
            mind_threshold (float): Individual missingness threshold
            maf_threshold (float): Minor allele frequency threshold
            biallelic_only (bool): use only biallelic alleles or not
            split_multi_allelic (bool): split multi-allelic alleles or not
            custom_args (str): Extra args to be used
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

        Args:
            script_file_list (list): list of script file paths
            output_file_name (str): output file path
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
        dsub_provider: str = "google-batch",
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
                provider=dsub_provider,
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
            job_limit: int = None,
            provider: str = "google-batch",
            region: str = "us-central1",
    ):
        if job_limit is None:
            job_limit = len(dsub_jobs)
        if show_all:
            dsub_user = os.getenv("OWNER_EMAIL").split("@")[0]
            command = f"dstat --provider {provider} --project $GOOGLE_PROJECT --location {region} --users {dsub_user} --status '*' --limit {job_limit}"
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