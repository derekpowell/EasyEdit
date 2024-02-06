import os
if os.path.isdir('/scratch/dmpowell'):
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/dmpowell/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/scratch/dmpowell/.cache/huggingface/datasets'
print(os.getenv('TRANSFORMERS_CACHE'))
print(os.getenv('HF_DATASETS_CACHE'))

import numpy as np
import torch
from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, GPT2LMHeadModel, AutoModelForCausalLM

import pandas as pd
import json
import janitor

from easyeditor.util import nethook
from easyeditor.custom import * # gets my custom functions

# from easyeditor.editors import LOG
# import logging
# LOG.setLevel(logging.ERROR) # stops cluttering up notebook

import torch.nn.functional as F

from contextlib import redirect_stdout

device = torch.device("cuda")

## --- load data

baseline_df, edits_df, eval_df = load_data()

prefix_fwd, prefix_rev = load_prefixes(verbose = False)

## --- set up test mode (or not)
MODE = "testing" #"testing"
if MODE=="testing":
    import random
    random.seed(123)
    edits_df = edits_df.sample(100)

## -- set up models and do edits with different methods

hparam_config = dict()
results = dict()

hparam_config["ROME"] = {"HyperParams": ROMEHyperParams, "path": 'hparams/ROME/llama-7b.yaml', "edit_method": "ROME"}
hparam_config["ICE"] = {"HyperParams": ROMEHyperParams, "path": 'hparams/ROME/llama-7b.yaml', "edit_method": "ICE"}
hparam_config["FT"] = {"HyperParams": FTHyperParams, "path": 'hparams/FT/llama-7b.yaml', "edit_method": "FT"}
# hparam_config["PMET"] = {"HyperParams": PMETHyperParams, "path": 'hparams/PMET/llama-7b.yaml', "edit_method": "PMET"} # broken
# hparam_config["GRACE"] = {"HyperParams": GraceHyperParams, "path": 'hparams/GRACE/llama-7B.yaml', "edit_method": "GRACE"} # broken


for edit_method, HPARAMS in hparam_config.items():    

    hparams = HPARAMS["HyperParams"].from_hparams(HPARAMS["path"])
    
    # with OutputLogger("my_log", "INFO") as redirector:
    edited_model = EditedModel(hparams, auth_token())
    res = edit_and_evaluate(edits_df, eval_df, edited_model, edit_method, prefix_fwd = "", prefix_rev = "", log_file = "results/log-2024-02-05.txt")
    
    res.to_csv("results/csv/" + hparams.model_name.replace("/", "-") + "-" + edit_method +  ".csv")

    results[HPARAMS["edit_method"]] = res