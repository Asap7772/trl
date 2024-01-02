for i in $(seq 0 5); do
    echo $i
    sbatch /iris/u/asap7772/trl/examples/anikait_dev/scripts/run_sbatch.sh $i
done