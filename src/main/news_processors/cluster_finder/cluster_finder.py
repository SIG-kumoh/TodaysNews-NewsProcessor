import logging
from numpy import dot
from numpy.linalg import norm
from datetime import date, datetime, timedelta

from persistence.repository import *
from persistence.models import Cluster, PreprocessedCluster, RelatedCluster

from module.scheduler import Schedule


class ClusterFinder(Schedule):
    def __init__(self, threshold: float = 4):
        self._cluster_repository = ClusterRepository()
        self._preprocessed_cluster_repository = PreprocessedClusterRepository()
        self._related_cluster_repository = RelatedClusterRepository()

        self._threshold: float = threshold

        self.logger = logging.getLogger('cluster finder')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __call__(self, t_date: date = date.today()):
        self.logger.info("relational cluster find start")

        clusters: list[Cluster] = self._cluster_repository.find_all_by_duration(t_date)
        related_clusters: list[RelatedCluster] = []

        for cluster in clusters:
            limit_date = datetime.combine(cluster.regdate.date(), datetime.min.time()) - timedelta(weeks=2)

            raw_related_clusters = self.find_related_cluster(target_cluster=cluster, limit_date=limit_date)
            for raw_related_cluster in raw_related_clusters:
                related_clusters.append(RelatedCluster(parent_cluster_id=cluster.cluster_id,
                                                       child_cluster_id=raw_related_cluster.cluster_id))

        related_clusters = self._related_cluster_repository.insert(related_clusters)
        for cluster in clusters:
            for related_cluster in related_clusters:
                if related_cluster.parent_cluster_id == cluster.related_cluster_id:
                    cluster.related_cluster_id = related_cluster.related_cluster_id
        self._cluster_repository.insert(clusters)

        self.logger.info("relational cluster find finished")

    def find_related_cluster(self, target_cluster: Cluster, limit_date: date) -> list[Cluster]:
        """연관 클러스터 탐색
        주어진 클러스터와 가장 유사한 클러스터를 추출한다.
        기준 클러스터의 임베딩과 DB에서 조회한 클러스터의 임베딩의 코사인 유사도를 측정하여 유사도 스코어를 산출한다.
        기준 클러스터의 토픽과 DB에서 조회한 클러스터의 토픽과의 일치도를 비교하여 토픽 스코어를 산출한다.
        두 스코어의 합을 기준으로 연관 클러스터를 판단한다.

        :param target_cluster: 기준이 될 클러스터
        :param limit_date: DB에서 조회할 날짜 제한(금일부터 limit_date까지만을 조회)
        :return: 기준 클러스터와 가장 유사하다고 판단되는 클러스터 모델 (최대 3개)
        """
        duration: tuple[date, datetime] = (limit_date, target_cluster.regdate - timedelta(days=1))
        preprocessed_target: PreprocessedCluster = self._preprocessed_cluster_repository.find_all_by_cluster(target_cluster)[0]
        clusters: list[Cluster] = self._cluster_repository.find_all_by_section_id_and_duration(section_id=target_cluster.section_id,
                                                                                               duration=duration)
        preprocessed_clusters: list[PreprocessedCluster] = self._preprocessed_cluster_repository.find_all_by_cluster(clusters=clusters)
        relational_clusters: list[tuple[float, Cluster]] = []

        for cluster, preprocessed_cluster in zip(clusters, preprocessed_clusters):
            embedding = preprocessed_cluster.embedding
            words = preprocessed_cluster.words
            sim_score: float = self._cal_cos_sim(preprocessed_target.embedding, embedding)
            topic_score: int = self._is_matching(preprocessed_target.words, words)
            # topic_score의 최대치는 3, sim_score의 최대치는 1이므로 sim_score에 3 가중치 설정

            relational_clusters.append((sim_score * 3 + topic_score, cluster))

        relational_clusters = sorted(relational_clusters, key=lambda x: x[0], reverse=True)
        result_list: list[Cluster] = []
        for idx, element in enumerate(relational_clusters):
            if idx > 2:
                break

            score, cluster = element
            if score > self._threshold:
                result_list.append(cluster)

        return result_list

    def _is_matching(self, topics1: list[str], topics2: list[str]) -> int:
        count = 0
        for topic1 in topics1:
            for topic2 in topics2:
                if topic1 in topic2 or topic2 in topic1:
                    count = count + 1

        return count

    def _cal_cos_sim(self, a, b) -> float:
        return dot(a, b) / (norm(a) * norm(b))
