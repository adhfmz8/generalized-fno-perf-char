#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=fno_sanity_check
#SBATCH --output=logs/sanity_check-%j.out
#SBATCH --error=logs/sanity_check-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- Setup ---
module load python
module load cudatoolkit
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
BENCHMARK_SCRIPT="$SLURM_SUBMIT_DIR/benchmark_v2.py"

echo ">>> Starting FNO vs TFNO Sanity Check"
echo ">>> Goal: Demonstrate parameter explosion in 3D Dense FNO"

# Create a temporary file to hold the comparison table
RESULTS_FILE="fno_vs_tfno_results.txt"
echo "Model,Res,Batch,Latency(ms),Mem(MB),Params(M)" > $RESULTS_FILE

# --- Function to run and parse specific metrics ---
run_check() {
    MODEL=$1
    RES=$2
    BATCH=$3
    
    echo "---------------------------------------------------"
    echo "Running: $MODEL 3D @ Res $RES, Batch $BATCH"
    
    # Run the benchmark
    OUTPUT=$($PY_EXEC $BENCHMARK_SCRIPT \
        --model $MODEL \
        --dim 3 \
        --res $RES \
        --batch $BATCH \
        --width 64 \
        --modes 16 \
        --precision tf32 \
        --unroll 10 2>&1)
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "FAILED (Likely OOM)"
        echo "$MODEL,$RES,$BATCH,FAILED,FAILED,FAILED" >> $RESULTS_FILE
    else
        # Parse the CSV line from the output
        # CSV Format from benchmark_v2.py:
        # RESULT,Model,Dim,Res,Batch,Prec,Lat,Thr,Mem,Comp,Data,Params,GFlops,TFlops
        
        CSV_LINE=$(echo "$OUTPUT" | grep "RESULT,")
        
        # Extract relevant fields using cut (comma delimiter)
        # 2: Model, 4: Res, 5: Batch, 7: Latency, 9: Mem, 12: Params
        LATENCY=$(echo $CSV_LINE | cut -d',' -f7)
        MEM=$(echo $CSV_LINE | cut -d',' -f9)
        PARAMS=$(echo $CSV_LINE | cut -d',' -f12)
        
        echo "  > Latency: ${LATENCY} ms"
        echo "  > Memory:  ${MEM} MB"
        echo "  > Params:  ${PARAMS} M"
        
        echo "$MODEL,$RES,$BATCH,$LATENCY,$MEM,$PARAMS" >> $RESULTS_FILE
    fi
}

# --- 1. The Champion: TFNO (Our Baseline) ---
run_check "TFNO" 64 1

# --- 2. The Challenger: Dense FNO (Same Config) ---
# Warning: This might OOM or be extremely heavy
run_check "FNO" 64 1

# --- 3. The Fallback: Dense FNO (Smaller Config) ---
# If the above fails, this shows that even at small scale it is heavy
run_check "FNO" 32 1

echo "---------------------------------------------------"
echo ">>> SUMMARY RESULTS <<<"
column -t -s "," $RESULTS_FILE
echo "---------------------------------------------------"