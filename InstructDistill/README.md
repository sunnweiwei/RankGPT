# Instruction Distillation

Code for paper [Instruction Distillation Makes LLMs Efficient Pointwise Rankers]()

This project aims to improve the efficiency of LLMs as rankers via instruction distillation.

## Pre-trained Models

| Model | Link |
| ---- | ---- |
| Rank-Flan-T5-XL | |
| Rank-Flan-T5-Large | |
| Rank-Flan-T5-Base | |
| Rank-LLaMA-2-7B | |

The following code show how to predict the relevance of a paired (query, passage).

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

query = "How much impact do masks have on preventing the spread of the COVID-19?"
passage = "Title: Universal Masking is Urgent in the COVID-19 Pandemic: SEIR and Agent Based Models, Empirical Validation, Policy Recommendations Content: We present two models for the COVID-19 pandemic predicting the impact of universal face mask wearing upon the spread of the SARS-CoV-2 virus--one employing a stochastic dynamic network based compartmental SEIR (susceptible-exposed-infectious-recovered) approach, and the other employing individual ABM (agent-based modelling) Monte Carlo simulation--indicating (1) significant impact under (near) universal masking when at least 80% of a population is wearing masks, versus minimal impact when only 50% or less of the population is wearing masks, and (2) significant impact when universal masking is adopted early, by Day 50 of a regional outbreak, versus minimal impact when universal masking is adopted late. These effects hold even at the lower filtering rates of homemade masks. To validate these theoretical models, we compare their predictions against a new empirical data set we have collected"
instrcution = "Predict whether the given passage answer the question.\n\nQuestion: {0}\n\nPassage: {1}\n\nDoes the passage answer the question?"
instrcution = instrcution.format(query, passage)
```
Use case of flan-t5 models
```python
tokenizer = AutoTokenizer.from_pretrained("fireballoon/rank-flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("fireballoon/rank-flan-t5-xl", torch_dtype=torch.float16)
token_of_Yes = 2163
features = tokenizer([instrcution,], padding=True, truncation=True, return_tensors="pt", max_length=1024)
features['decoder_input_ids'] = torch.zeros(len(batch), 1).long()
scores = model(**features).logits[:, -1, token_of_Yes]
```
Use case of llama models
```python
tokenizer = AutoTokenizer.from_pretrained("fireballoon/rank-llama-2-7b", use_fast=False, padding_side="left")
model = AutoModelForCausalLM.from_pretrained("fireballoon/rank-llama-2-7b", torch_dtype=torch.float16)
token_of_Yes = 3869
features = tokenizer([instrcution,], padding=True, truncation=True, return_tensors="pt", max_length=1024)
scores = model(**features).logits[:, -1, token_of_Yes]
```

## Training
Retrieve passage using BM25
```
python bm25_retrieval.py
```
(optional) Evaluating Pairwise Ranking Prompting (PRP) on benchmarks.
```
python pairwise_ranking.py --model google/flan-t5-xl --eval true --generate false
```
Getting predictions of PRP on MS MARCO (`data/marco-train-10k.jsonl`, can be downloaded from [RankGPT](https://github.com/sunnweiwei/RankGPT/tree/main#download-data-and-model)). The ranking results will be saved at `out/marco-train-10k-flan-xl.json`. 
```
python pairwise_ranking.py \
--model google/flan-t5-xl \
--eval false \
--generate true \
--data data/marco-train-10k.jsonl \
--save_path out/marco-train-10k-flan-xl.json
```
Training the pointwise ranker using PRP's predictions. The model checkpoints well be saved at `out/rank-flan-t5-xl`.
```
python instruction_distill.py \
--model google/flan-t5-xl \
--loss rank_net \
--data data/marco-train-10k.jsonl \
--save_path out/rank-flan-t5-xl \
--permutation out/marco-train-10k-flan-xl.json \
--do_train true \
--do_eval false
```
Converting deepspeed checkpoint.
```
python
```
### 



