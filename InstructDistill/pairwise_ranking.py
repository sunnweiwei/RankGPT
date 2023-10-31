# Pariwise Ranking Prompting
#

import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import os
import argparse


FLAN_PRP_PROMPT = '''Question: Given a query "{0}", which of the following two passages is more relevant to the query?

passage A: {1}

passage B: {2}

Output the identifier of the more relevant passage. The answer must be passage A or passage B.
Answer:'''

GPT_PRP_PROMPT = '''### System:
You are a pairwise passage ranker that can judge which passages is more relevant to the query.

### User:
Given a query "{0}", which of the following two passages is more relevant to the query?

Passage A: {1}

Passage B: {2}

Output the identifier of the more relevant passage. The answer must be Passage A or Passage B.

### Assistant:
The more relevant passage is Passage'''


def eval_prp(model_name):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from trec_eval import EvalFunction
    from bm25_retrieval import THE_RESULTS

    print(model_name)

    if 't5' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        token_passage = 5454
        token_A = 71
        token_B = 272
        PROMPT = FLAN_PRP_PROMPT
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left",
                                                  model_max_length=4096)
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        token_passage = None
        token_A = 319
        token_B = 350
        PROMPT = GPT_PRP_PROMPT

    model = model.cuda()
    model.eval()

    # data_list = ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']
    data_list = ['dl19', 'dl20']
    for data_name in data_list:
        print()
        print('#' * 20)
        print(f'Now eval [{data_name}]')
        print('#' * 20)

        rank_results = json.load(open(THE_RESULTS[data_name]))
        saved = []
        for item in tqdm(rank_results):
            q = item['query']
            passages = [psg['content'] for i, psg in enumerate(item['hits'])][:100]
            passages = [' '.join(psg.split()[:100]) for psg in passages]
            if len(passages) == 0:
                saved.append('')
                continue

            all_score = [0 for _ in range(len(passages))]

            new_passages = []
            for i in range(len(passages)):
                for j in range(len(passages)):
                    if i == j:
                        continue
                    prompt = PROMPT.format(q, passages[i], passages[j])
                    new_passages.append([prompt, i, j])
            passages = new_passages

            i = 0
            while i < len(passages):
                batch = passages[i: i + 10]
                i += 10
                features = tokenizer([psg[0] for psg in batch], padding=True, truncation=True, return_tensors="pt")
                if 't5' in model_name:
                    features['decoder_input_ids'] = torch.tensor([[0, token_passage]] * len(batch)).long()
                features = {k: v.cuda() for k, v in features.items()}
                with torch.no_grad():
                    scores = model(**features).logits[:, -1]
                for score, psg in zip(scores, batch):
                    if score[token_A] > score[token_B]:
                        all_score[psg[1]] += 1
                    elif score[token_B] > score[token_A]:
                        all_score[psg[2]] += 1
                    else:
                        all_score[psg[1]] += 0.5
                        all_score[psg[2]] += 0.5
            all_score = [s + 1 / (10 + r) for r, s in enumerate(all_score)]
            ranked = np.argsort(all_score)[::-1]
            response = ' > '.join([str(ss + 1) for ss in ranked])
            saved.append(response)

        rank_results = EvalFunction.receive_responses(rank_results, saved, cut_start=0, cut_end=100)
        tmp_path = 'tmp_rank_results'
        EvalFunction.write_file(rank_results, tmp_path)

        EvalFunction.main(data_name, tmp_path)


def generate_data(model_name, data_path, save_path):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if 't5' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        token_passage = 5454
        token_A = 71
        token_B = 272
        PROMPT = FLAN_PRP_PROMPT
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left",
                                                  model_max_length=4096)
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        token_passage = None
        token_A = 319
        token_B = 350
        PROMPT = GPT_PRP_PROMPT

    rank_results = [json.loads(line) for line in open(data_path)][:10000]
    saved = []
    for item in tqdm(rank_results):
        q = item['query']
        passages = [psg['text'] for i, psg in enumerate(item['retrieved_passages'])][:20]
        passages = [' '.join(psg.split()[:100]) for psg in passages]
        if len(passages) == 0:
            saved.append('')
            continue

        all_score = [0 for _ in range(len(passages))]

        new_passages = []
        for i in range(len(passages)):
            for j in range(len(passages)):
                if i == j:
                    continue
                prompt = PROMPT.format(q, passages[i], passages[j])
                new_passages.append([prompt, i, j])
        passages = new_passages

        i = 0
        while i < len(passages):
            batch = passages[i: i + 10]
            i += 10
            features = tokenizer([psg[0] for psg in batch], padding=True, truncation=True, return_tensors="pt",
                                 max_length=1024)
            if 't5' in model_name:
                features['decoder_input_ids'] = torch.tensor([[0, token_passage]] * len(batch)).long()
            features = {k: v.cuda() for k, v in features.items()}
            with torch.no_grad():
                scores = model(**features).logits[:, -1]
            for score, psg in zip(scores, batch):
                if score[token_A] > score[token_B]:
                    all_score[psg[1]] += 1
                elif score[token_B] > score[token_A]:
                    all_score[psg[2]] += 1
                else:
                    all_score[psg[1]] += 0.5
                    all_score[psg[2]] += 0.5
        all_score = [s + 1 / (10 + r) for r, s in enumerate(all_score)]
        ranked = np.argsort(all_score)[::-1]
        response = ' > '.join([str(ss + 1) for ss in ranked])
        saved.append(response)
    json.dump(saved, open(save_path, 'w'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/flan-t5-xl')
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--generate', type=bool, default=True)
    parser.add_argument('--data', type=str, default='data/marco-train-10k.jsonl')
    parser.add_argument('--save_path', type=str, default='out/rpr-flan-t5-xl.json')
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == '__main__':
    args = parse_args()

    # Eval pairwise ranking on benchmarks
    if args.eval:
        eval_prp(args.model)

    # Get predictions on MS MARCO
    if args.generate:
        generate_data(args.model, args.data, args.save_path)
