THE_RESULTS = {
    'dl19': 'data/rank_results/dl19.json',
    'dl20': 'data/rank_results/dl20.json',
    'covid': 'data/rank_results/beir-trec-covid.json',
    'arguana': 'data/rank_results/beir-arguana.json',
    'touche': 'data/rank_results/beir-touche.json',
    'news': 'data/rank_results/beir-news.json',
    'scifact': 'data/rank_results/beir-scifact.json',
    'fiqa': 'data/rank_results/beir-fiqa.json',
    'scidocs': 'data/rank_results/beir-scidocs.json',
    'nfc': 'data/rank_results/beir-nfc.json',
    'quora': 'data/rank_results/beir-quora.json',
    'dbpedia': 'data/rank_results/beir-dbpedia.json',
    'fever': 'data/rank_results/beir-fever.json',
    'robust04': 'data/rank_results/beir-robust04.json',
    'signal': 'data/rank_results/beir-signal.json',
}

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
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',

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

from pyserini.search import LuceneSearcher, get_topics, get_qrels
import json
from tqdm import tqdm


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks

def do_retrieval():
    for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
        print('#' * 20)
        print(f'Evaluation on {data}')
        print('#' * 20)

        # Retrieve passages using pyserini BM25.
        # Get a specific doc:
        # * searcher.num_docs
        # * json.loads(searcher.object.reader.document(4).fields[1].fieldsData) -> {"id": "1", "contents": ""}
        try:
            searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
            topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
            qrels = get_qrels(THE_TOPICS[data])
            rank_results = run_retriever(topics, searcher, qrels, k=100)

            # Store JSON in rank_results to a file
            with open(f'rank_results_{data}.json', 'w') as f:
                json.dump(rank_results, f, indent=2)
            # Store the QRELS of the dataset
            with open(f'qrels_{data}.json', 'w') as f:
                json.dump(qrels, f, indent=2)
        except:
            print(f'Failed to retrieve passages for {data}')

    for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw',
                 'mrtydi-te', 'mrtydi-th']:
        print('#' * 20)
        print(f'Evaluation on {data}')
        print('#' * 20)

        # Retrieve passages using pyserini BM25.
        try:
            searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
            topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
            qrels = get_qrels(THE_TOPICS[data])
            rank_results = run_retriever(topics, searcher, qrels, k=100)
            rank_results = rank_results[:100]

            # Store JSON in rank_results to a file
            with open(f'data/rank_results/{data}.json', 'w') as f:
                json.dump(rank_results, f, indent=2)
            # Store the QRELS of the dataset
            with open(f'data/qrels/{data}.json', 'w') as f:
                json.dump(qrels, f, indent=2)
        except:
            print(f'Failed to retrieve passages for {data}')
