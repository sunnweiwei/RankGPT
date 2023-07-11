# RankGPT: LLMs as Re-Ranking Agent

Code for paper "[Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent](https://arxiv.org/abs/2304.09542)"

This project aims to explore generative LLMs such as ChatGPT and GPT-4 for relevance ranking in Information Retrieval (IR).

## News
- **[2023.07.11]** Release a new test set NovelEval with the novel search questions and passages that have not been contaminated by the latest LLMs (e.g., GPT-4). See [NovelEval](https://github.com/sunnweiwei/RankGPT/tree/main/NovelEval) for details.
- **[2023.04.23]** Sharing 100K ChatGPT predicted permutations on MS MARCO training set [here](#download-data-and-model).
- **[2023.04.19]** Our paper is now available at https://arxiv.org/abs/2304.09542

## Quick example
Below defines a query and three candidate passages:

```python
item = {
    'query': 'How much impact do masks have on preventing the spread of the COVID-19?',
    'hits': [
        {'content': 'Title: Universal Masking is Urgent in the COVID-19 Pandemic: SEIR and Agent Based Models, Empirical Validation, Policy Recommendations Content: We present two models for the COVID-19 pandemic predicting the impact of universal face mask wearing upon the spread of the SARS-CoV-2 virus--one employing a stochastic dynamic network based compartmental SEIR (susceptible-exposed-infectious-recovered) approach, and the other employing individual ABM (agent-based modelling) Monte Carlo simulation--indicating (1) significant impact under (near) universal masking when at least 80% of a population is wearing masks, versus minimal impact when only 50% or less of the population is wearing masks, and (2) significant impact when universal masking is adopted early, by Day 50 of a regional outbreak, versus minimal impact when universal masking is adopted late. These effects hold even at the lower filtering rates of homemade masks. To validate these theoretical models, we compare their predictions against a new empirical data set we have collected'},
        {'content': 'Title: Masking the general population might attenuate COVID-19 outbreaks Content: The effect of masking the general population on a COVID-19 epidemic is estimated by computer simulation using two separate state-of-the-art web-based softwares, one of them calibrated for the SARS-CoV-2 virus. The questions addressed are these: 1. Can mask use by the general population limit the spread of SARS-CoV-2 in a country? 2. What types of masks exist, and how elaborate must a mask be to be effective against COVID-19? 3. Does the mask have to be applied early in an epidemic? 4. A brief general discussion of masks and some possible future research questions regarding masks and SARS-CoV-2. Results are as follows: (1) The results indicate that any type of mask, even simple home-made ones, may be effective. Masks use seems to have an effect in lowering new patients even the protective effect of each mask (here dubbed"one-mask protection") is'},
        {'content': 'Title: To mask or not to mask: Modeling the potential for face mask use by the general public to curtail the COVID-19 pandemic Content: Face mask use by the general public for limiting the spread of the COVID-19 pandemic is controversial, though increasingly recommended, and the potential of this intervention is not well understood. We develop a compartmental model for assessing the community-wide impact of mask use by the general, asymptomatic public, a portion of which may be asymptomatically infectious. Model simulations, using data relevant to COVID-19 dynamics in the US states of New York and Washington, suggest that broad adoption of even relatively ineffective face masks may meaningfully reduce community transmission of COVID-19 and decrease peak hospitalizations and deaths. Moreover, mask use decreases the effective transmission rate in nearly linear proportion to the product of mask effectiveness (as a fraction of potentially infectious contacts blocked) and coverage rate (as'}
    ]
}

```

We can re-rank the passages using ChatGPT with instructional permutation generation:

```python
from rank_gpt import permutation_pipeline
new_item = permutation_pipeline(item, rank_start=0, rank_end=3, model_name='gpt-3.5-turbo', openai_key='Your OPENAI Key!')
print(new_item)
```

We get the following result:

```python
{
    'query': 'How much impact do masks have on preventing the spread of the COVID-19?',
    'hits': [
        {'content': 'Title: Universal Masking is Urgent in the COVID-19 Pandemic: SEIR and Agent Based Models, Empirical Validation, Policy Recommendations Content: We present two models for the COVID-19 pandemic predicting the impact of universal face mask wearing upon the spread of the SARS-CoV-2 virus--one employing a stochastic dynamic network based compartmental SEIR (susceptible-exposed-infectious-recovered) approach, and the other employing individual ABM (agent-based modelling) Monte Carlo simulation--indicating (1) significant impact under (near) universal masking when at least 80% of a population is wearing masks, versus minimal impact when only 50% or less of the population is wearing masks, and (2) significant impact when universal masking is adopted early, by Day 50 of a regional outbreak, versus minimal impact when universal masking is adopted late. These effects hold even at the lower filtering rates of homemade masks. To validate these theoretical models, we compare their predictions against a new empirical data set we have collected'},
        {'content': 'Title: To mask or not to mask: Modeling the potential for face mask use by the general public to curtail the COVID-19 pandemic Content: Face mask use by the general public for limiting the spread of the COVID-19 pandemic is controversial, though increasingly recommended, and the potential of this intervention is not well understood. We develop a compartmental model for assessing the community-wide impact of mask use by the general, asymptomatic public, a portion of which may be asymptomatically infectious. Model simulations, using data relevant to COVID-19 dynamics in the US states of New York and Washington, suggest that broad adoption of even relatively ineffective face masks may meaningfully reduce community transmission of COVID-19 and decrease peak hospitalizations and deaths. Moreover, mask use decreases the effective transmission rate in nearly linear proportion to the product of mask effectiveness (as a fraction of potentially infectious contacts blocked) and coverage rate (as'},
        {'content': 'Title: Masking the general population might attenuate COVID-19 outbreaks Content: The effect of masking the general population on a COVID-19 epidemic is estimated by computer simulation using two separate state-of-the-art web-based softwares, one of them calibrated for the SARS-CoV-2 virus. The questions addressed are these: 1. Can mask use by the general population limit the spread of SARS-CoV-2 in a country? 2. What types of masks exist, and how elaborate must a mask be to be effective against COVID-19? 3. Does the mask have to be applied early in an epidemic? 4. A brief general discussion of masks and some possible future research questions regarding masks and SARS-CoV-2. Results are as follows: (1) The results indicate that any type of mask, even simple home-made ones, may be effective. Masks use seems to have an effect in lowering new patients even the protective effect of each mask (here dubbed"one-mask protection") is'}
    ]
}
```

<details>
<summary>Step by step example</summary>
  
  ```python
  from rank_gpt import create_permutation_instruction, run_llm, receive_permutation
  
  # (1) Create permutation generation instruction
  messages = create_permutation_instruction(item=item, rank_start=0, rank_end=3, model_name='gpt-3.5-turbo')
  # (2) Get ChatGPT predicted permutation
  permutation = run_llm(messages, openai_key="Your OPENAI Key!", model_name=model_name='gpt-3.5-turbo')
  # (3) Use permutation to re-rank the passage
  item = receive_permutation(item, permutation, rank_start=0, rank_end=3)
  
  ```
  
</details>

## Sliding window strategy

We introduce a sliding window strategy for the instructional permutation generation, that enables LLMs to rank more passages than their maximum token limit.

The idea is to rank from back to front using a sliding window, re-ranking only the passages within the window at a time.

Below is an example by re-ranking 3 passages with window size of 2 and step size of 1:

```python
from rank_gpt import sliding_windows
new_item = sliding_windows(item, rank_start=0, rank_end=3, window_size=2, step=1, model_name='gpt-3.5-turbo', openai_key='Your OPENAI Key!')
print(new_item)
```

## Evaluation on Benchmarks
We use [pyserini](https://github.com/castorini/pyserini) to retrieve 100 passages for each query and re-rank them using instructional permutation generation.

Example of evaluation on TREC-DL19:

```python
from pyserini.search import LuceneSearcher, get_topics, get_qrels
from rank_gpt import run_retriever, sliding_windows
import tempfile
openai_key = None  # Your openai key

# Retrieve passages using pyserini BM25.
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
topics = get_topics('dl19-passage')
qrels = get_qrels('dl19-passage')
rank_results = run_retriever(topics, searcher, qrels, k=100)

# Run sliding window permutation generation
new_results = []
for item in tqdm(rank_results):
    new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo', openai_key=openai_key)
    new_results.append(new_item)

# Evaluate nDCG@10
from trec_eval import EvalFunction
temp_file = tempfile.NamedTemporaryFile(delete=False).name
write_eval_file(new_results, temp_file)
EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', 'dl19-passage', temp_file])
```

Run evaluation on all benchmarks

```sh
python run_evaluation.py
```

Below are the results (average nDCG@10) of our preliminary experiments on [TREC](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html), [BEIR](https://github.com/beir-cellar/beir) and [Mr. TyDi](https://github.com/castorini/mr.tydi).

![Results on benchmarks](assets/benchmark-results.png)

## Installation

The installation process assumes that you use Python 3.9, or that you install a `conda` environment with Python 3.9.

```shell
# Create a conda environment (optional)
conda create -n rankgpt python=3.9
conda activate rankgpt
```

Install Java 11 to use `pyserini`

```shell
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt install openjdk-11-jdk
```

Install `faiss` for GPU or CPU

```shell
# Install with GPU support
conda install faiss-gpu pytorch pytorch-cuda -c pytorch -c nvidia
# Or, if just CPU
conda install -c pytorch faiss-cpu
```

Install other dependencies

```shell
pip install -r requirements.txt
```

Finally, add your OpenAI API key as envionment variable

```shell
export OPENAI_KEY=YOUR_OPENAI_KEY
```

## Training Specialized Models

### Download data and model

|   File  | Note | Link |
|:-------------------------------|:--------|:--------:|
| marco-train-10k.jsonl | 10K queries sampled from MS MARCO | [Google drive](https://drive.google.com/file/d/1G3MpQ5a4KgUS13JJZFE9aQvCbQfgSQzj/view?usp=share_link) |
| marco-train-10k-gpt3.5.json |  Permutations predicted by ChatGPT   | [Google drive](https://drive.google.com/file/d/1i7ckK7kN7BAqq5g7xAd0dLv3cTYYiclA/view?usp=share_link) |
| deberta-10k-rank_net    |  Specialized Deberta model trained with RankNet loss | [Google drive](https://drive.google.com/file/d/1-KEpJ2KnJCqiJof4zNEA4m78tnwgxKhb/view?usp=share_link)  |
|marco-train-100k.jsonl | 100K queries from MS MARCO | [Google drive](https://drive.google.com/file/d/1OgF4rj89FWSr7pl1c7Hu4x0oQYIMwhik/view?usp=share_link) |
| marco-train-100k-gpt3.5.json | Permutations by ChatGPT of the 100K queries | [Google drive](https://drive.google.com/file/d/1z327WOKr70rC4UfOlQVBQnuLxChi_uPs/view?usp=share_link) |

### Distill LLM to a small specialized model

```bash
python specialization.py \
--model microsoft/deberta-v3-base \
--loss rank_net \
--data data/marco-train-10k.jsonl \
--permutation marco-train-10k-gpt3.5.json \
--save_path out/deberta-10k-rank_net \
--do_train true \
--do_eval true
```

or run on multi-gpus, using [accelerate](https://github.com/huggingface/accelerate):

```bash
accelerate launch --num_processes 4 specialization.py \
--model microsoft/deberta-v3-base \
--loss rank_net \
--data data/marco-train-10k.jsonl \
--permutation marco-train-10k-gpt3.5.json \
--save_path out/deberta-10k-rank_net \
--do_train true \
--do_eval true
```

### Evaluate the distilled model on benchmarks

```bash
python specialization.py \
--model out/deberta-10k-rank_net \
--do_train false \
--do_eval true
```

The following figure show the results of distilled specialized model with different model size and number of training queires.

![Specialization results.](assets/specialization-results.png)

## Cite

```latex
@article{Sun2023IsCG,
  title={Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent},
  author={Weiwei Sun and Lingyong Yan and Xinyu Ma and Pengjie Ren and Dawei Yin and Zhaochun Ren},
  journal={ArXiv},
  year={2023},
  volume={abs/2304.09542}
}
```
