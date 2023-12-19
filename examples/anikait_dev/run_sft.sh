export PJRT_DEVICE=TPU
export WANDB_PROJECT=sft_tpu
# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
export PYTHONPATH=/home/anikaitsingh/trl:$PYTHONPATH
python /home/anikaitsingh/trl/examples/anikait_dev/sft.py \
--dataset_path "tatsu-lab/alpaca_farm" \
--pretrained_dir="microsoft/phi-1_5" \
--output_dir="/home/anikaitsingh/trl/exp_checkpoints/phi-1_5b_sft" \
