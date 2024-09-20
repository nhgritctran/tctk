import os
import subprocess


def _dsub_script(
        machine_type,
        disk_type,
        docker_image,
        boot_disk_size=50,
        disk_size=256,

):
    script = """
    dsub \
    --provider google-cls-v2 \
    --machine-type "${MACHINE_TYPE}" \
    --disk-type "pd-ssd" \
    --boot-disk-size 50 \
    --disk-size 256 \
    --user-project "${GOOGLE_PROJECT}" \
    --project "${GOOGLE_PROJECT}" \
    --image "phetk/phetk:0.2.1rc5p" \
    --network "network" \
    --subnetwork "subnetwork" \
    --service-account "$(gcloud config get-value account)" \
    --user "${DSUB_USER_NAME}" \
    --logging "${WORKSPACE_BUCKET}/dsub/logs/{job-name}/{user-id}/$(date +'%Y%m%d')/{job-id}-{task-id}-{task-attempt}.log" \
    "$@" \
    --name "${JOB_NAME}" \
    --env GOOGLE_PROJECT=${GOOGLE_PROJECT} \
    --input COHORT_CSV_PATH=${COHORT_CSV_PATH} \
    --input PHECODE_COUNT_CSV_PATH=${PHECODE_COUNT_CSV_PATH} \
    --output OUTPUT_FILE="${WORKSPACE_BUCKET}/dsub/results/${JOB_NAME}/${USER_NAME}/$(date +'%Y%m%d')/${OUTPUT_FILE}" \
    --output LOG_FILE="${WORKSPACE_BUCKET}/dsub/results/${JOB_NAME}/${USER_NAME}/$(date +'%Y%m%d')/${LOG_FILE}" \
    --script ${SCRIPT_NAME}
    """


def run_dsub():
    pass
