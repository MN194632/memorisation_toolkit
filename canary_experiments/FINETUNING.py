#reference inspiration: https://huggingface.co/docs/transformers/tasks/asr
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer, Wav2Vec2ForCTC,Wav2Vec2Processor,Wav2Vec2CTCTokenizer
import argparse
import sys
import canary_experiments.CONFIG as CONFIG
from dataloader_libri import LibriSpeechDataset
from dataset_utils import DataCollatorCTCWithPadding
from dataloader_canaries import CanariesDataset
from torchmetrics.text import CharErrorRate
import numpy as np
from functools import partial
from canary_frequency_coordinator import CanaryFrequencyCoordinator
from evaluate import load
from utils import save_config_as_json,print_trainable_parameters,freeze_all_except_head,get_config_class

def main():
    parser = argparse.ArgumentParser(description='Train ASR model with specified config')
    parser.add_argument('--config', type=str, default='Config', 
                        help='Config class name to use (e.g., Config15, Config2)')
    
    args = parser.parse_args()
    
    # Get the specified config
    config = get_config_class(args.config)
    print(f"Using config: {args.config}")
    print(f"Output directory: {config.output_dir}")
    
    save_config_as_json(config, config.output_dir)
    data_collator = DataCollatorCTCWithPadding(processor=config.processor, padding=True)
    processor = config.processor

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=8,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=config.max_steps, 
        gradient_checkpointing=True,
        fp16=True,
        group_by_length=True,
        eval_strategy="steps",
        lr_scheduler_type="cosine", 
        per_device_eval_batch_size=8,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        )

    coordinator = CanaryFrequencyCoordinator(config, frequencies=getattr(config, 'frequencies', [1, 2, 4, 8, 16]))
    train_dataset  = coordinator.create_mixed_dataset(only_canaries=config.only_canaries)
    dev_dataset = LibriSpeechDataset(config.dev_csv, config.base_audio_dir,config.processor,config.tokenizer)

    cer = load("cer")
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        cer_score = cer.compute(predictions=pred_str, references=label_str)

        return {"cer": cer_score}

    if config.freeze:
        model = config.model
        model = freeze_all_except_head(model)
        print_trainable_parameters(model)
        model.train()
    else:
        model = config.model
        model.train()
        print_trainable_parameters(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=dev_dataset, 
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

    print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best CER score: {trainer.state.best_metric}")

if __name__ == "__main__":
    main()