# NovelEval
*A new test set with the novel queries and passages that have not been contaminated by the latest LLMs*

The questions in the current benchmark dataset are typically gathered years ago, which raises the issue that existing LLMs already possess knowledge of these questions.
Furthermore, since many LLMs do not disclose information about their training data, there is a potential risk of contamination of the existing benchmark test set.
However, re-ranking models are expected to possess the capability to comprehend, deduce, and rank knowledge that is inherently unknown to them.

Therefore, we suggest constructing **continuously updated IR test sets** to ensure that the questions, passages to be ranked, and relevance annotations have not been learned by the latest LLMs for a fair evaluation.

## Data Collection
As an initial effort, we built **NovelEval-2306**, a novel test set with 21 novel questions collected during 2023-06. 
This test set is constructed by gathering questions and passages fromfrom 4 domains that were published after the release of GPT-4.
To ensure that GPT-4 did not possess prior knowledge of these questions, we presented them to both gpt-4-0314 and gpt-4-0613.
For instance, question *"Which film was the 2023 Palme d'Or winner?"* pertains to the Cannes Film Festival that took place on May 27, 2023, rendering its answer inaccessible to most existing LLMs.
Next, we searched 20 candidate passages for each question using Google search.
The relevance of these passages was manually labeled as: 0 for not relevant, 1 for partially relevant, and 2 for relevant.


## Files
| Type | Filename | Format|
| ---- | ---- | ---- |
| Corpus | [corpus.tsv](https://github.com/sunnweiwei/RankGPT/blob/main/NovelEval/corpus.tsv) | tsv: docid, content |
| Queries | [queries.tsv](https://github.com/sunnweiwei/RankGPT/blob/main/NovelEval/queries.tsv) | tsv: qid, query |
| Qrels | [qrels.txt](https://github.com/sunnweiwei/RankGPT/blob/main/NovelEval/qrels.txt) | TREC qrels format: qid, Q0, docid, relevance-score |

## Results

| Method | nDCG@1 | nDCG@5 | nDCG@10 |
| ---- | ----- | ----- | ----- |
| BM25 | 33.33 | 45.96 | 55.77 |
| monoBERT (340M) | 78.57 | 70.65 | 77.27 |
| monoT5 (220M) | 83.33 | 77.46 | 81.27 |
| monoT5 (3B) | 83.33 | 78.38 | 84.62 |
| gpt-3.5-turbo | 76.19 | 74.15 | 75.71 |
| **gpt-4** | **85.71** | **87.49** | **90.45** |
