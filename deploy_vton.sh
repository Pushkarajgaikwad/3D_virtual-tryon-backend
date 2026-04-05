#!/bin/bash
#SBATCH --job-name=vton_api
#SBATCH --output=logs/vton_%j.out
#SBATCH --error=logs/vton_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
#  3D Virtual Try-On API — SLURM Deployment Script
#  Launches Redis + FastAPI/Uvicorn on a GPU node
# ============================================================

set -euo pipefail

PROJECT_DIR="/scratch/nidhi.raut.aissmsioit/3D_virtual-tryon-backend"
cd "${PROJECT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "============================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : $(hostname)"
echo "  GPUs        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Start Time  : $(date)"
echo "============================================"

# ----------------------------------------------------------
# 1. Activate Conda Environment
# ----------------------------------------------------------
# Source conda — adjust the path if your conda install differs
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "[ERROR] Could not find conda.sh — please update the path in this script."
    exit 1
fi

conda activate vton
echo "[✓] Conda environment 'vton' activated"
echo "    Python: $(which python) — $(python --version)"

# ----------------------------------------------------------
# 2. Start Redis Server (background / daemonized)
# ----------------------------------------------------------
REDIS_SERVER="${PROJECT_DIR}/redis-stable/src/redis-server"

if [ ! -x "${REDIS_SERVER}" ]; then
    echo "[ERROR] redis-server not found or not executable at ${REDIS_SERVER}"
    echo "        Run 'make' inside redis-stable/ first."
    exit 1
fi

"${REDIS_SERVER}" --daemonize yes \
                  --bind 127.0.0.1 \
                  --port 6379 \
                  --dir "${PROJECT_DIR}" \
                  --logfile "${PROJECT_DIR}/logs/redis_${SLURM_JOB_ID}.log"

echo "[✓] Redis server started (daemonized on 127.0.0.1:6379)"

# Quick health check
sleep 1
if "${PROJECT_DIR}/redis-stable/src/redis-cli" ping | grep -q "PONG"; then
    echo "    Redis PING → PONG ✓"
else
    echo "[WARN] Redis did not respond to PING — check logs/redis_${SLURM_JOB_ID}.log"
fi

# ----------------------------------------------------------
# 3. Start FastAPI Server
# ----------------------------------------------------------
echo "[→] Starting Uvicorn (FastAPI) on 0.0.0.0:8000 ..."
echo "    Access via: http://$(hostname):8000"
echo "    Docs:       http://$(hostname):8000/docs"
echo "============================================"

# Run in foreground so SLURM tracks the process lifetime
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info

# ----------------------------------------------------------
# Cleanup on exit
# ----------------------------------------------------------
echo "[→] Shutting down Redis ..."
"${PROJECT_DIR}/redis-stable/src/redis-cli" shutdown nosave 2>/dev/null || true
echo "[✓] Job completed at $(date)"
