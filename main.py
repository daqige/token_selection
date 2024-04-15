#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

'''
This file is modified from the HuggingFace official example repository:
https://github.com/huggingface/transformers/blob/v4.21.1/examples/pytorch/text-classification/run_glue.py
All the modifications:
    1. Add VcasSampleArguments to the argument parser
    2. Initialize VcasSampleScheme with VcasSampleArguments
    3. Process the original model with VcasModelProcessor
    4. Substitute the original Trainer with VcasTrainer
Please search for "CHANGES" to see the modifications in detail.
'''

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_from_disk

from models.modeling_gpt2 import  GPT2LMHeadModel
from models.configuration_gpt2 import GPT2Config

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,

    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling
)


from vcas import VcasSampleArguments, VcasSampleScheme, VcasModelProcessor, VcasTrainer

logger = logging.getLogger(__name__)





def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # CHANGES #1: Add VcasSampleArguments to the argument parser 
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser(VcasSampleArguments)


    sample_args = parser.parse_args_into_dataclasses()




    # Set seed before initializing model.
    set_seed(1)

    
    tokenizer = AutoTokenizer.from_pretrained("./gpt2_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024


    tokenized_datasets = load_from_disk("/code/remote/data/tokenized_openweb")
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=5000)
    
    config = GPT2Config(n_head=12,n_embd=768,n_positions=1024,n_layer=12, resid_pdrop=0,
                        embd_pdrop=0, attn_pdrop=0)
    model = GPT2LMHeadModel(config)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # CHANGES #2: Initialize VcasSampleScheme with VcasSampleArguments
    sample_scheme = VcasSampleScheme(sample_args)

    # CHANGES #3: Process the original model with VcasModelProcessor
    from models.modeling_gpt2 import GPT2Block

    processor = VcasModelProcessor(model, GPT2Block, sample_scheme)
    processor.process()

    print(model)


    training_args = TrainingArguments(output_dir='./ckpts/test',
                                    run_name='test',
                                    per_device_train_batch_size = 4,
                                    per_device_eval_batch_size = 4,
                                    num_train_epochs=10,
                                    save_steps = 10000,
                                    evaluation_strategy='steps',
                                    eval_steps = 1000,
                                    logging_steps=5, 
                                    learning_rate=6e-4, #6e-4
                                    adam_epsilon = 1e-4,
                                    gradient_accumulation_steps=16,
                                    save_safetensors=True,
                                    bf16=True)
    
    trainer = VcasTrainer(
        model=model,
        sample_scheme=sample_scheme,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        metrics["train_samples"] = len(tokenized_datasets["train"])

        trainer.save_model()  # Saves the tokenizer too for easy upload


        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()





if __name__ == "__main__":
    main()