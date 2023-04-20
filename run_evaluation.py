THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04-flat',
    'signal': 'beir-v1.0.0-signal1m-flat',

    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}

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

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}

from rank_gpt import run_retriever, sliding_windows, write_eval_file
from pyserini.search import LuceneSearcher, get_topics, get_qrels
from tqdm import tqdm
import tempfile

openai_key = None  # Your openai key

for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
    print('#' * 20)
    print(f'Evaluation on {data}')
    print('#' * 20)

    # Retrieve passages using pyserini BM25.
    searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
    topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
    qrels = get_qrels(THE_TOPICS[data])
    rank_results = run_retriever(topics, searcher, qrels, k=100)

    # Run sliding window permutation generation
    new_results = []
    for item in tqdm(rank_results):
        new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
                                   model_name='gpt-3.5-turbo', openai_key=openai_key)
        new_results.append(new_item)

    # Evaluate nDCG@10
    from trec_eval import EvalFunction

    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(new_results, temp_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file])


for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw', 'mrtydi-te', 'mrtydi-th']:
    print('#' * 20)
    print(f'Evaluation on {data}')
    print('#' * 20)

    # Retrieve passages using pyserini BM25.
    searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])[:100]
    topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
    qrels = get_qrels(THE_TOPICS[data])
    rank_results = run_retriever(topics, searcher, qrels, k=100)

    # Run sliding window permutation generation
    new_results = []
    for item in tqdm(rank_results):
        new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
                                   model_name='gpt-3.5-turbo', openai_key=openai_key)
        new_results.append(new_item)

    # Evaluate nDCG@10
    from trec_eval import EvalFunction

    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(new_results, temp_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file])

