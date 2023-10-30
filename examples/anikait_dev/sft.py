print('Starting imports')

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from absl import app, flags
print('Done with imports')

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, 'the output directory')
flags.DEFINE_integer('batch_size', 4, 'the batch size')
flags.DEFINE_float('learning_rate', 8e-6, 'the learning rate')
flags.DEFINE_float('weight_decay', 0, 'the weight decay')
flags.DEFINE_integer('num_train_epochs', 50, 'the number of training epochs')
flags.DEFINE_integer('max_steps', -1, 'the number of training steps')
flags.DEFINE_bool('push_to_hub', False, 'Push the model to HF Hub')
flags.DEFINE_string('hub_model_id', None, 'The name of the model on HF Hub')
flags.DEFINE_integer("gradient_accumulation_steps", 1, "Gradient accumulation steps")
flags.DEFINE_bool("gradient_checkpointing", False, "Whether to use gradient checkpointing")
flags.DEFINE_bool("mixed_precision", False, "Whether to use mixed precision")
flags.DEFINE_integer("max_seq_length", 512, "The maximum sequence length")

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'

def main(_):
    dataset = load_dataset("tatsu-lab/alpaca_farm", split="sft")

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            inst, inp, out = example['instruction'][i], example['input'][i], example['output'][i]
            if inp:
                text = f"{PROMPT_TOKEN}{inst}\n{inp}{eos}{ASSISTANT_TOKEN}{out}{eos}"
            else:
                text = f"{PROMPT_TOKEN}{inst}{eos}{ASSISTANT_TOKEN}{out}{eos}"
            output_texts.append(text)
        return output_texts

    response_template = ASSISTANT_TOKEN
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    extra_kwargs = {}
    if FLAGS.output_dir is not None:
        extra_kwargs['output_dir'] = FLAGS.output_dir

    training_args = TrainingArguments(
        do_predict=True,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        push_to_hub=FLAGS.push_to_hub,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        gradient_checkpointing=FLAGS.gradient_checkpointing,
        fp16=FLAGS.mixed_precision,
        logging_first_step=True,
        optim="adafactor",
        report_to='wandb',
        hub_model_id=FLAGS.hub_model_id,
        per_device_train_batch_size=FLAGS.batch_size,
        per_device_eval_batch_size=FLAGS.batch_size,
        num_train_epochs=FLAGS.num_train_epochs,
        **extra_kwargs
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=FLAGS.max_seq_length,
    )

    trainer.train() 
    
if __name__ == "__main__":
    app.run(main)