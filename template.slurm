#!/bin/sh
#SBATCH --account=studiegrupper-cogito        # <<< ENDRE HER til din konto på Idun
#SBATCH --job-name="ksim_zeroth-nr1"          # <<< ENDRE HER: beskrivende navn for jobben
#SBATCH --time=06:00:00                        # <<< ENDRE HER: estimert kjøretid (HH:MM:SS)
#SBATCH --partition=GPUQ                        # <<< ENDRE HER: partisjon; GPUQ hvis GPU, ellers small/medium
#SBATCH --gres=gpu:1                            # <<< ENDRE HER: antall GPUer, fjern hvis CPU-only
#SBATCH --mem=40GB                               # <<< ENDRE HER: minne som trengs
#SBATCH --nodes=1
#SBATCH --constraint="(a100|h100|h200)&(gpu40g|gpu80g)"  # <<< ENDRE HER: spesifikke maskiner/GPUer hvis ønskelig
#SBATCH --output=slurm_outputs/train_output_%j.txt
#SBATCH --error=slurm_outputs/train_error_%j.txt
#SBATCH --mail-user=adriajl@ntnu.no            # <<< ENDRE HER: din epost
#SBATCH --mail-type=ALL

# ------------------------------
# Info om jobben
# ------------------------------
echo "Running from directory: $WORKDIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"

# ------------------------------
# Opprett output-mappe
# ------------------------------
mkdir -p slurm_outputs

# ------------------------------
# Conda / Python miljø
# ------------------------------
ENV_PATH="/cluster/work/$USER/ksim_env/"      # <<< ENDRE HER: sti til miljøet på Idun

module purge
module load Anaconda3/2024.02-1              # <<< ENDRE HER: last inn riktig Python-modul

# Lag miljø hvis det ikke finnes
if [ ! -d "$ENV_PATH" ]; then
  conda create -y --prefix ${ENV_PATH} python=3.11
fi

source activate ${ENV_PATH}

# ------------------------------
# Installer dependencies via Poetry
# ------------------------------
# Installer Poetry hvis det ikke finnes
pip install --user poetry

# Installer alle dependencies definert i pyproject.toml
poetry install

echo "Installed dependencies:"
pip freeze

# ------------------------------
# GPU / MuJoCo (valgfritt)
# ------------------------------
# export MUJOCO_GL=osmesa  # <<< Aktiver kun hvis prosjektet bruker MuJoCo med headless rendering

# ------------------------------
# Kjør script
# ------------------------------
python train.py  # <<< ENDRE HER: bytt til scriptet du vil kjøre, f.eks. train_standup.py

# ------------------------------
# Eventuell håndtering av output (valgfritt)
# ------------------------------
# Du kan legge til kode her hvis du vil flytte output-filer til spesifikke mapper

# ------------------------------
# Avslutt miljø
# ------------------------------
conda deactivate
echo "Job finished"
