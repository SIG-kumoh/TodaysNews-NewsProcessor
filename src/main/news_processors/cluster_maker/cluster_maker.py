import logging
import warnings

from persistence.models import Cluster, PreprocessedCluster, Article
from ._ctfidf import ClassTfidfTransformer
from ._clustering import clustering
from persistence.repository import *

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import numpy as np
import pandas as pd

from tqdm import tqdm
from umap import UMAP
from typing import List, Dict, Union
from datetime import datetime, date, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from module.scheduler import Schedule
from module.summarizer.multi_docs_summarizer import MultiDocsSummarizer, Centroid

CONFIG_PATH = 'resources/config/cluster_maker/config.yml'


class ClusterMaker(Schedule):
    def __init__(self):
        self.umap = UMAP(n_neighbors=15,
                         n_components=5,
                         min_dist=0.0,
                         metric='cosine',
                         low_memory=False)
        self.vectorizer = CountVectorizer(tokenizer=_no_process, preprocessor=_no_process, token_pattern=None)
        self.ctfidf = ClassTfidfTransformer()
        self.mds = MultiDocsSummarizer(SentenceTransformer('jhgan/ko-sroberta-nli'))

        self.article_repository = ArticleRepository()
        self.preprocessed_article_repository = PreprocessedArticleRepository()
        self.cluster_repository = ClusterRepository()
        self.preprocessed_cluster_repository = PreprocessedClusterRepository()

        self.noise_threshold = 0.5
        self.min_document = 20

        self.section_id = {}
        self.load_section()

        # TODO 리팩터링
        self.logger = logging.getLogger('cluster')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __call__(self, t_date: date):
        if t_date is None:
            t_date = date.today()

        # 당일 클러스터 삭제 - TODO 오류있음
        clusters = self.cluster_repository.find_all_by_duration(t_date)
        self.cluster_repository.delete(clusters)

        # 섹션 별 클러스터링
        for section_name in self.section_id.keys():
            self.clustering(section_name, t_date)

    def clustering(self, section_name: str, t_date: date):
        t_datetime = datetime.combine(t_date, datetime.min.time())
        article_list = self.article_repository.find_all_by_section_id(section_id=self.section_id[section_name],
                                                                      t_date=t_date)

        if self.min_document < len(article_list):
            topic_words, labels = self._topic_clustering(article_list)

            centroids = self._extract_centroids(article_list, labels, topic_words)
            labeled_clusters = self._make_labeled_clusters(labels=labels,
                                                           t_datetime=t_datetime,
                                                           centroids=centroids,
                                                           section_name=section_name)

            self.cluster_repository.insert(list(labeled_clusters.values()))
            labeled_preprocessed_clusters = self._make_preprocessed_clusters(labels=labels,
                                                                             labeled_clusters=labeled_clusters,
                                                                             centroids=centroids,
                                                                             topic_words=topic_words)
            self.preprocessed_cluster_repository.insert(list(labeled_preprocessed_clusters.values()))

            for label, article in zip(labels, article_list):
                if label == -1:
                    article.cluster_id = None
                else:
                    article.cluster_id = labeled_clusters[label].cluster_id

            article_list = self.article_repository.update(article_list)

            # TODO 삭제
            # _save_file(article_list, labels, topic_words, section_name, t_date)

        else:
            self.logger.debug(f'{section_name} section, too small')

    def load_section(self):
        section_repository = SectionRepository()
        section_list = section_repository.find_all()
        for section in section_list:
            self.section_id[section.section_name] = section.section_id

    def _extract_centroids(self,
                           article_list,
                           labels,
                           topics: Dict[int, list[tuple[str, float]]]) -> Dict[int, Centroid]:
        centroids = {}
        for label, article in zip(labels, article_list):
            if label not in centroids:
                centroids[label] = []
            centroids[label].append(article)

        for label, articles_list_in_cluster in tqdm(centroids.items()):
            topic_words = []
            for topics_in_cluster in topics[label]:
                topic_words.append(topics_in_cluster[0])

            preprocessed_list = self.preprocessed_article_repository.find_all_by_article(articles=articles_list_in_cluster)
            centroids[label] = self.mds.summarize(article_list=articles_list_in_cluster,
                                                  preprocessed_list=preprocessed_list,
                                                  topics=topic_words)

        return centroids

    def _make_labeled_clusters(self,
                               labels,
                               t_datetime,
                               centroids: Dict[int, Centroid],
                               section_name):
        labeled_clusters = {}
        for label in labels:
            if label != -1:
                new_topic = Cluster(regdate=t_datetime,
                                    img_url=centroids[label].article.img_url,
                                    title=centroids[label].article.title,
                                    summary=centroids[label].summary,
                                    section_id=self.section_id[section_name],
                                    related_cluster_id=None)
                labeled_clusters[label] = new_topic

        return labeled_clusters

    def _make_preprocessed_clusters(self,
                                    labels,
                                    labeled_clusters,
                                    centroids: Dict[int, Centroid],
                                    topic_words):
        labeled_preprocessed_clusters = {}
        for label in labels:
            if label != -1:
                preprocessed_article = self.preprocessed_article_repository.find_all_by_article(centroids[label].article)
                new_pre_cluster = PreprocessedCluster(cluster_id=labeled_clusters[label].cluster_id,
                                                      embedding=preprocessed_article[0].embedding,
                                                      words=[e[0] for e in topic_words[label]])
                labeled_preprocessed_clusters[label] = new_pre_cluster

        return labeled_preprocessed_clusters

    def _topic_clustering(self, article_list):
        # 1. embeddings, tokens 리스트 생성
        preprocessed_list = self.preprocessed_article_repository.find_all_by_article(articles=article_list)
        embeddings = []
        tokens_list = []
        for e in preprocessed_list:
            embeddings.append(e.embedding)
            tokens_list.append(e.tokens)

        # 2. UMAP 알고리즘 사용하여 차원축소
        reduced_embeddings = np.nan_to_num(self.umap.fit_transform(embeddings))

        # 3. HDBSCAN, Mean-Shift 사용하여 클러스터링
        labels = clustering(reduced_embeddings)

        # 4. 각 군집에 대하여 c-TF-IDF로 토픽 추출
        classed_tokens = _tokens_per_label(labels, tokens_list)
        c_tf_idf, c_words = self._extract_topic(classed_tokens)
        topics = _extract_words_per_topic(c_tf_idf, c_words, labels, 10)

        # 5. 토픽으로 노이즈 제거
        tf_idf, words = self._extract_topic(tokens_list)
        topics = self._remove_noise_topics(topics, labels, tf_idf)
        labels = self._remove_noise_news(topics, labels, tf_idf)

        # 6. 노이즈 제거된 군집에 대하여, c-TF-IDF로 다시 토픽 추출
        classed_tokens = _tokens_per_label(labels, tokens_list)
        c_tf_idf, c_words = self._extract_topic(classed_tokens)
        topics = _extract_words_per_topic(c_tf_idf, c_words, labels, 10)

        return topics, labels

    def _extract_topic(self, tokens_list):
        self.vectorizer.fit(tokens_list)
        X = self.vectorizer.transform(tokens_list)
        words = self.vectorizer.get_feature_names_out()

        ctfidf = self.ctfidf.fit(X)
        c_tf_idf = ctfidf.transform(X)

        return c_tf_idf, words

    def _remove_noise_topics(self, topics, labels, matrix):
        reprocessed_topics = {}
        for label in topics.keys():
            reprocessed_topics[label] = {}
            for word, tf_idf in topics[label]:
                reprocessed_topics[label][word] = 0.0

        for label, tf_idf in zip(labels, matrix):
            cur_topics = reprocessed_topics[label]
            for word in cur_topics.keys():
                if word != '':
                    cur_topics[word] += tf_idf.toarray()[0][self.vectorizer.vocabulary_[word]]

        for label in topics.keys():
            avg_tf_idf = sum(reprocessed_topics[label].values()) / len(reprocessed_topics[label])
            for idx in list(reversed(range(len(topics[label])))):
                word = topics[label][idx][0]
                if reprocessed_topics[label][word] < avg_tf_idf:
                    del topics[label][idx]

        return topics

    def _remove_noise_news(self, topics, labels, matrix):
        tf_idf_values = []
        thresholds = {}

        for label, tf_idf in zip(labels, matrix):
            if label not in thresholds:
                thresholds[label] = []

            cur_topics = topics[label]
            cur_tf_idf = 0.0

            for word, _ in cur_topics:
                if word != '':
                    cur_tf_idf += tf_idf.toarray()[0][self.vectorizer.vocabulary_[word]]

            tf_idf_values.append(cur_tf_idf)
            thresholds[label].append(cur_tf_idf)

        for label in thresholds.keys():
            series = pd.Series(thresholds[label])
            q1 = series.quantile(.25)
            q3 = series.quantile(.75)
            iqr = q3 - q1
            thresholds[label] = q1 - iqr * 1.3

        return labels


def _no_process(e):
    return e


def _top_n_idx_sparse(matrix, n: int) -> np.ndarray:
    indices = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)
    return np.array(indices)


def _top_n_values_sparse(matrix, indices: np.ndarray) -> np.ndarray:
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
        top_values.append(scores)
    return np.array(top_values)


def _extract_words_per_topic(c_tf_idf, words: List[str], labels, top_n: int = 30):
    labels = sorted(list(set(labels)))
    indices = _top_n_idx_sparse(c_tf_idf, top_n)
    scores = _top_n_values_sparse(c_tf_idf, indices)
    sorted_indices = np.argsort(scores, 1)
    indices = np.take_along_axis(indices, sorted_indices, axis=1)
    scores = np.take_along_axis(scores, sorted_indices, axis=1)

    topics = {label: [(words[word_index], score)
                      if word_index is not None and score > 0 else ("", 0.00001)
                      for word_index, score in zip(indices[index][::-1], scores[index][::-1])]
              for index, label in enumerate(labels)}
    return topics


def _tokens_per_label(labels, tokens_list):
    labeled_tokens = pd.DataFrame(columns=['Label', 'Tokens'], data=zip(labels, tokens_list))
    tokens_per_label = labeled_tokens.groupby(['Label'], as_index=False).agg({'Tokens': 'sum'})
    classed_tokens = tokens_per_label.Tokens.values
    return classed_tokens


def _save_file(news_list, labels, topic_words, section, t_date):
    distributed_docs = []
    for doc, tag in zip(news_list, labels):
        distributed_docs.append({'tag': tag,
                                 'id': doc.article_id,
                                 'title': doc.title,
                                 'url': doc.url,
                                 'content': doc.content})
    distributed_docs.sort(key=lambda x: x['tag'])

    print('save to csv...')
    topic_df = pd.DataFrame(columns=['topic_words'], index=list(topic_words.keys()),
                            data=[str([e2[0] for e2 in e1]) for e1 in topic_words.values()])
    distributed_docs_df = pd.DataFrame(columns=['tag', 'id', 'title', 'url', 'content'], data=distributed_docs)

    topic_df.to_csv(f'../temp/{section}_topic_list_{str(t_date)}.csv', encoding='utf-8')
    distributed_docs_df.to_csv(f'../temp/{section}_distributed_docs_{str(t_date)}.csv', encoding='utf-8')