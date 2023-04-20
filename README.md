# RankGPT: Passage Re-Ranking with ChatGPT

This project aims to explore generative LLMs such as ChatGPT and GPT-4 for relevance ranking in Information Retrieval (IR).

We aim to answer the following two questions: 
<ol>
  <li> How does ChatGPT perform on passage re-ranking tasks? </li>
  <li> How to distill the ranking capabilities of ChatGPT to a smaller, specialized model? </li>
</ol>

To answer the first question, we introduce an instructional permutation generation appraoch to instruct LLMs to directly output the permutations of a group of passages.

To answer the second question, we train a cross-encoder using 10K ChatGPT predicted permutations on MS MARCO.

Below are the results (average nDCG@10) of our preliminary experiments on TREC, BEIR and Mr. TyDi.

![Results on benchmarks](assets/results.png)
