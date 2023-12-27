# -*- coding: utf-8 -*-
# @Time    : 12/21/2023 1:58 AM
# @Author  : yuzhn
# @File    : save_embed.py
# @Software: PyCharm
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

arguments = argparse.ArgumentParser()
arguments.add_argument("--model", default = "NousResearch/Llama-2-70b-hf")
arguments.add_argument("--bad_embed_path",
                       default = "")
args = arguments.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
bad_embed_param = torch.load(os.path.join(args.bad_embed_path, "bad_embedding.pt")).to(model.device)
bad_index = torch.load(os.path.join(args.bad_embed_path, "bad_index.pt"))
embedding_params = model.get_input_embeddings()
embedding_params.weight.data[bad_index] = bad_embed_param.to(embedding_params.weight.dtype)
model.set_input_embeddings(embedding_params)
model.save_pretrained(os.path.join(args.bad_embed_path, "good_model"), max_shard_size = "10GB")
tokenizer.save_pretrained(os.path.join(args.bad_embed_path, "good_model"))
print("done")
