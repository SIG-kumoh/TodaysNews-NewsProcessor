import logging
import warnings

from persistence.models import Cluster, PreprocessedCluster, Article, HotCluster
from ._ctfidf import ClassTfidfTransformer
from ._clustering import clustering
from persistence.repository import *
from .cluster_finder import ClusterFinder

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import numpy as np
import pandas as pd

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
        self.cluster_finder = ClusterFinder()

        self.article_repository = ArticleRepository()
        self.preprocessed_article_repository = PreprocessedArticleRepository()
        self.cluster_repository = ClusterRepository()
        self.preprocessed_cluster_repository = PreprocessedClusterRepository()
        self.hot_cluster_repository = HotClusterRepository()

        self.noise_threshold = 0.5
        self.min_cluster_size = 3
        self.min_document = 20

        self.section_id = {}
        self.load_section()

        # TODO 리팩터링
        self.logger = logging.getLogger('cluster')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __call__(self, t_date: date = None):
        self.logger.info("clustering start")

        if t_date is None:
            t_date = date.today()

        for section_name in self.section_id.keys():
            self.clustering(section_name, t_date)
        self.cluster_finder(t_date)

        # 핫 클러스터 생성
        new_hot_clusters = self.make_hot_cluster(t_date)
        old_hot_clusters = self.hot_cluster_repository.find_all_by_duration(t_date)
        self.hot_cluster_repository.delete(old_hot_clusters)
        self.hot_cluster_repository.insert(new_hot_clusters)

        self.logger.info("clustering finished")

    def clustering(self, section_name: str, t_date: date):
        t_datetime = datetime.combine(t_date, datetime.min.time())
        article_list = self.article_repository.find_all_by_section_id(section_id=self.section_id[section_name],
                                                                      t_date=t_date)

        if self.min_document < len(article_list):
            topic_words, labels = self._topic_clustering(article_list)
            topic_words, labels = self._remove_minimum_cluster(topic_words, labels)
            centroids = self._extract_centroids(article_list, labels, topic_words)
            labeled_clusters = self._make_labeled_clusters(labels=labels,
                                                           t_datetime=t_datetime,
                                                           centroids=centroids,
                                                           section_name=section_name)

            # 클러스터 교체
            clusters = self.cluster_repository.find_all_by_section_id(section_id=self.section_id[section_name],
                                                                      duration=t_date)
            self.cluster_repository.delete(clusters)
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

        topics[-1] = [('temp', 0)]
        for label, articles_list_in_cluster in centroids.items():
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
                                    centroid_id=centroids[label].article.article_id,
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
        topics = _extract_words_per_topic(c_tf_idf, c_words, labels, 5)

        # 5. 토픽으로 노이즈 제거
        topics = self._remove_noise_topics(topics, article_list, labels)
        labels = self._remove_noise_articles(topics, article_list, labels)

        # 6. 노이즈 제거된 군집에 대하여, c-TF-IDF로 다시 토픽 추출
        classed_tokens = _tokens_per_label(labels, tokens_list)
        c_tf_idf, c_words = self._extract_topic(classed_tokens)
        topics = _extract_words_per_topic(c_tf_idf, c_words, labels, 3)

        return topics, labels

    def _extract_topic(self, tokens_list):
        self.vectorizer.fit(tokens_list)
        X = self.vectorizer.transform(tokens_list)
        words = self.vectorizer.get_feature_names_out()

        ctfidf = self.ctfidf.fit(X)
        c_tf_idf = ctfidf.transform(X)

        return c_tf_idf, words

    def _remove_noise_topics(self, topics, article_list: list[Article], labels):
        reprocessed_topics = {}
        for label in topics.keys():
            reprocessed_topics[label] = {}
            for word, tf_idf in topics[label]:
                reprocessed_topics[label][word] = 0.0

        for cluster_idx, article in zip(labels, article_list):
            cur_topics = reprocessed_topics[cluster_idx]
            for word in cur_topics.keys():
                if word != '':
                    word_content_frequency = article.content.count(word)
                    word_title_frequency = article.title.count(word)
                    cur_topics[word] += (word_title_frequency * 2 + word_content_frequency)

        for label in topics.keys():
            threshold = sum(reprocessed_topics[label].values()) / len(reprocessed_topics[label])
            for idx in list(reversed(range(len(topics[label])))):
                word = topics[label][idx][0]
                if reprocessed_topics[label][word] <= threshold:
                    del topics[label][idx]

        return topics

    def _remove_noise_articles(self, topics, article_list: list[Article], labels):
        score_list = []
        thresholds = {}

        for article_idx in range(labels.size):
            cluster_idx = labels[article_idx]
            cluster_topics = topics[cluster_idx]
            article = article_list[article_idx]
            cur_score = 0

            if cluster_idx not in thresholds:
                thresholds[cluster_idx] = []

            for word, _ in cluster_topics:
                if word != '':
                    word_content_frequency = article.content.count(word)
                    word_title_frequency = article.title.count(word)
                    cur_score += (word_title_frequency * 2 + word_content_frequency)

            score_list.append(cur_score)
            thresholds[cluster_idx].append(cur_score)

        for cluster_idx in thresholds.keys():
            threshold = (sum(thresholds[cluster_idx]) / len(thresholds[cluster_idx])) * 0.5
            thresholds[cluster_idx] = threshold

        labels_idx = 0
        for label, score in zip(labels, score_list):
            if thresholds[label] >= score:
                labels[labels_idx] = -1
            labels_idx += 1

        return labels

    def _remove_minimum_cluster(self, topic_words, labels):
        count = {key: 0 for key in topic_words.keys()}
        for label in labels:
            count[label] += 1

        delete_labels = []
        for key, count in count.items():
            if count < self.min_cluster_size:
                delete_labels.append(key)

        for idx in range(len(labels)):
            if labels[idx] in delete_labels:
                labels[idx] = -1

        return topic_words, labels

    def make_hot_cluster(self, t_date: date) -> List[HotCluster]:
        clusters = self.cluster_repository.find_all_by_duration(duration=t_date)
        counted_clusters = []
        for cluster in clusters:
            counted_clusters.append((cluster, self.article_repository.count_by_cluster_id(cluster.cluster_id)))
        counted_clusters = sorted(counted_clusters, key=lambda e: e[1], reverse=True)

        hot_clusters = []
        for idx in range(10):
            if idx >= len(counted_clusters):
                break
            cur_cluster = counted_clusters[idx][0]
            cur_count = counted_clusters[idx][1]

            hot_cluster = HotCluster(
                cluster_id=cur_cluster.cluster_id,
                regdate=cur_cluster.regdate,
                size=cur_count,
                namespace='test'
            )
            hot_clusters.append(hot_cluster)

        return hot_clusters


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
