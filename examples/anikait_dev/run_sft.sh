export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH
python /iris/u/asap7772/trl/examples/anikait_dev/sft.py \
--dataset_path "tatsu-lab/alpaca_farm" \
--pretrained_dir="EleutherAI/pythia-1.4b" \
--output_dir="/iris/u/asap7772/trl/exp_checkpoints/mistral_test_tpu"