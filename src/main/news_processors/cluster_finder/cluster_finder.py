import logging
from numpy import dot
from numpy.linalg import norm
from datetime import date, datetime, timedelta

from persistence.repository import *
from persistence.models import Cluster, PreprocessedCluster, RelatedCluster

from module.scheduler import Schedule


class ClusterFinder(Schedule):
    def __init__(self):
        self._cluster_repository = ClusterRepository()
        self._preprocessed_cluster_repository = PreprocessedClusterRepository()
        self._related_cluster_repository = RelatedClusterRepository()

        self.relational_cluster_threshold = 0.5

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
            limit_date = datetime.combine(date.today(), datetime.min.time()) - timedelta(weeks=4)

            self.logger.debug("find_relational_cluster method start")
            relational_clusters = self.find_relational_cluster(target_cluster=cluster, limit_date=limit_date)
            self.logger.debug("find_relational_cluster method end")
            for relational_cluster in relational_clusters:
                related_clusters.append(RelatedCluster(parent_cluster_id=cluster.cluster_id,
                                                       child_cluster_id=relational_cluster.cluster_id))

            self.logger.debug("find_time_series_cluster method start")
            time_series_clusters = self.find_time_series_cluster(target_cluster=cluster, limit_date=limit_date)
            self.logger.debug("find_time_series_cluster method end")
            for time_series_cluster in time_series_clusters:
                relational_clusters.append(RelatedCluster(parent_cluster_id=cluster.cluster_id,
                                                          child_cluster_id=time_series_cluster.cluster_id))
        self._related_cluster_repository.insert(related_clusters)

        self.logger.info("relational cluster find finished")

    def find_relational_cluster(self, target_cluster: Cluster, limit_date: date) -> list[Cluster]:
        """연관 클러스터 탐색
        주어진 클러스터와 가장 유사한 클러스터를 추출한다.
        기준 클러스터의 임베딩과 DB에서 조회한 클러스터의 임베딩의 코사인 유사도를 측정한다.
        임계값 이상인 클러스터를 연관 클러스터로 간주한다.

        :param target_cluster: 기준이 될 클러스터
        :param limit_date: DB에서 조회할 날짜 제한(금일부터 limit_date까지만을 조회)
        :return: 기준 클러스터와 가장 유사하다고 판단되는 클러스터 모델
        """
        target_embedding: list[str] = self._preprocessed_cluster_repository.find_all_by_cluster(target_cluster)[0].embedding
        clusters: list[Cluster] = self._cluster_repository.find_all_by_section_id_and_limit_date(section_id=target_cluster.section_id,
                                                                                                 limit_date=limit_date)
        preprocessed_clusters: list[PreprocessedCluster] = self._preprocessed_cluster_repository.find_all_by_cluster(clusters=clusters)
        relational_clusters: list[Cluster] = []

        for cluster, preprocessed_cluster in zip(clusters, preprocessed_clusters):
            embedding = preprocessed_cluster.embedding
            if self._cal_cos_sim(target_embedding, embedding) > self.relational_cluster_threshold:
                relational_clusters.append(cluster)

        # TODO 결과 체크 필요
        return relational_clusters

    def find_time_series_cluster(self, target_cluster: Cluster, limit_date: date) -> list[Cluster]:
        """시계열 클러스터 탐색
        주어진 클러스터와 동일한 토픽을 지닌 과거 클러스터를 추출한다.
        기준 클러스터의 토픽과 DB에서 조회한 클러스터의 토픽 중
        토픽 단어가 일치하는 것이 하나라도 존재하면 시계열 클러스터로 간주한다.

        :param target_cluster: 기준이 될 클러스터
        :param limit_date: DB에서 조회할 날짜 제한(금일부터 limit_date까지만을 조회)
        :return: 기준 클러스터의 시계열 클러스터
        """
        target_topic: list[str] = self._preprocessed_cluster_repository.find_all_by_cluster(target_cluster)[0].words
        clusters: list[Cluster] = self._cluster_repository.find_all_by_section_id_and_limit_date(section_id=target_cluster.section_id,
                                                                                                 limit_date=limit_date)
        preprocessed_clusters: list[PreprocessedCluster] = self._preprocessed_cluster_repository.find_all_by_cluster(clusters=clusters)
        time_series_cluster: list[Cluster] = []

        for cluster, preprocessed_cluster in zip(clusters, preprocessed_clusters):
            topic: list[str] = preprocessed_cluster.words
            if self._is_matching(target_topic, topic):
                time_series_cluster.append(cluster)

        # TODO 결과 체크 필요
        return time_series_cluster

    def _is_matching(self, topics1: list[str], topics2: list[str]) -> bool:
        for topic1 in topics1:
            for topic2 in topics2:
                if topic1 in topic2 or topic2 in topic1:
                    return True

        return False

    def _cal_cos_sim(self, a, b) -> float:
        return dot(a, b) / (norm(a) * norm(b))
