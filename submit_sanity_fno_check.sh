#!/bin/bash -l
#SBATCH --account=m4647
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --job-name=fno_scaling_killshot
#SBATCH --output=logs/killshot-%j.out
#SBATCH --error=logs/killshot-%j.err
#SBATCH -C 'gpu&hbm80g'

# --- Setup ---
module load python
module load cudatoolkit
PY_EXEC="$SCRATCH/no_paper_env/bin/python"
BENCHMARK_SCRIPT="$SLURM_SUBMIT_DIR/benchmark_v2.py"

echo ">>> Starting FNO vs TFNO Mode Scaling 'Kill Shot'"
echo ">>> Goal: Demonstrate Cubic Parameter Scaling in 3D Dense FNO"

# Create a clean results file
RESULTS_FILE="mode_scaling_results.csv"
echo "Model,Modes,Res,Latency(ms),Mem(MB),Params(M),Status" > $RESULTS_FILE

# --- Test Function ---
run_test() {
    MODEL=$1
    MODES=$2
    
    echo "---------------------------------------------------"
    echo "Running: $MODEL 3D @ Modes=$MODES (Res 64, Batch 1)"
    
    # We use '|| true' to prevent the script from exiting if FNO OOMs
    OUTPUT=$($PY_EXEC $BENCHMARK_SCRIPT \
        --model $MODEL \
        --dim 3 \
        --res 64 \
        --batch 1 \
        --width 64 \
        --modes $MODES \
        --precision tf32 \
        --unroll 10 2>&1)
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ">>> FAILED (Likely OOM)"
        echo "$MODEL,$MODES,64,0,0,0,FAILED" >> $RESULTS_FILE
    else
        # Parse output
        # RESULT format: RESULT,Model,Dim,Res,Batch,Prec,Lat,Thr,Mem,Comp,Data,Params...
        CSV_LINE=$(echo "$OUTPUT" | grep "RESULT,")
        
        # Extract fields
        LATENCY=$(echo $CSV_LINE | cut -d',' -f7)
        MEM=$(echo $CSV_LINE | cut -d',' -f9)
        PARAMS=$(echo $CSV_LINE | cut -d',' -f12)
        
        echo "  > Latency: ${LATENCY} ms"
        echo "  > Memory:  ${MEM} MB"
        echo "  > Params:  ${PARAMS} M"
        
        echo "$MODEL,$MODES,64,$LATENCY,$MEM,$PARAMS,SUCCESS" >> $RESULTS_FILE
    fi
}

# --- The Comparison Loop ---

# 1. Baseline (Modes 16)
run_test "TFNO" 16
run_test "FNO" 16

# 2. Mid-Range (Modes 24)
run_test "TFNO" 24
run_test "FNO" 24

# 3. High-Fidelity (Modes 32)
# Dense FNO params should explode here.
run_test "TFNO" 32
run_test "FNO" 32

echo "---------------------------------------------------"
echo ">>> FINAL SCALING TABLE <<<"
column -t -s "," $RESULTS_FILE
echo "---------------------------------------------------"