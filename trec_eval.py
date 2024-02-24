import pandas as pd
import tempfile
import os
import copy
from typing import Dict, Tuple
import pytrec_eval


def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


def get_qrels_file(name):
    THE_TOPICS = {
        'dl19': 'dl19-passage',
        'dl20': 'dl20-passage',
        'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test',
        'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test',
        'scidocs': 'beir-v1.0.0-scidocs-test',
        'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
        'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test',
        'signal': 'beir-v1.0.0-signal1m-test',
    }
    name = THE_TOPICS.get(name, '')
    name = name.replace('-test', '.test')
    name = 'data/label_file/qrels.' + name + '.txt'  # try to use cache
    if not os.path.exists():
        from pyserini.search import get_qrels_file
        return get_qrels_file(name)  # download from pyserini
    return name


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print('duplicate')
    return new_response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            try:
                new_response += str(int(c))
            except:
                new_response += ' '
    new_response = new_response.strip()
    return new_response


class EvalFunction:
    @staticmethod
    def receive_responses(rank_results, responses, cut_start=0, cut_end=100):
        print('receive_responses', len(responses), len(rank_results))
        for i in range(len(responses)):
            response = responses[i]
            response = clean_response(response)
            response = [int(x) - 1 for x in response.split()]
            response = remove_duplicate(response)
            cut_range = copy.deepcopy(rank_results[i]['hits'][cut_start: cut_end])
            original_rank = [tt for tt in range(len(cut_range))]
            response = [ss for ss in response if ss in original_rank]
            response = response + [tt for tt in original_rank if tt not in response]
            for j, x in enumerate(response):
                rank_results[i]['hits'][j + cut_start] = {
                    'content': cut_range[x]['content'], 'qid': cut_range[x]['qid'], 'docid': cut_range[x]['docid'],
                    'rank': cut_range[j]['rank'], 'score': cut_range[j]['score']}
        return rank_results

    @staticmethod
    def write_file(rank_results, file):
        print('write_file')
        with open(file, 'w') as f:
            for i in range(len(rank_results)):
                rank = 1
                hits = rank_results[i]['hits']
                for hit in hits:
                    f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                    rank += 1
        return True

    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        # print(qrels)
        run = pd.read_csv(run, delim_whitespace=True, header=None)
        qrels = pd.read_csv(qrels, delim_whitespace=True, header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep='\t', header=None, index=None)
        return temp_file

    @staticmethod
    def main(args_qrel, args_run):

        args_qrel = EvalFunction.trunc(args_qrel, args_run)

        assert os.path.exists(args_qrel)
        assert os.path.exists(args_run)

        with open(args_qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        with open(args_run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        all_metrics = trec_eval(qrel, run, k_values=(1, 5, 10))
        print(all_metrics)
        return all_metrics


if __name__ == '__main__':
    EvalFunction.main('dl19', 'ranking_results_file')
