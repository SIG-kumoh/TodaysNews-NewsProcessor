import logging
from datetime import date, datetime, timedelta

from persistence.repository import *
from persistence.models import Cluster, PreprocessedCluster

from module.scheduler import Schedule


class ClusterFinder(Schedule):
    def __init__(self):
        self._cluster_repository = ClusterRepository()
        self._preprocessed_cluster_repository = PreprocessedClusterRepository()

        self.logger = logging.getLogger('cluster finder')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __call__(self, t_date: date = date.today()):
        self.logger.info("relational cluster find start")

        clusters: list[Cluster] = self._cluster_repository.find_all_by_duration(t_date)
        updated_clusters: list[Cluster] = []

        for cluster in clusters:
            limit_date = datetime.combine(date.today(), datetime.min.time()) - timedelta(weeks=4)
            updated_clusters = (updated_clusters +
                                self.find_relational_cluster(cluster) +
                                self.find_time_series_cluster(target_cluster=cluster,
                                                              limit_date=limit_date))

        # TODO 연관 테이블에 삽입

        self.logger.info("relational cluster find finished")

    def find_relational_cluster(self, target_cluster: Cluster) -> list[Cluster]:
        """연관 클러스터 탐색
        주어진 클러스터와 가장 유사한 클러스터를 추출한다.

        :param target_cluster: 기준이 될 클러스터
        :return: 기준 클러스터와 가장 유사하다고 판단되는 클러스터 모델
        """

        # TODO 구체적인 로직 구상
        pass

    def find_time_series_cluster(self, target_cluster: Cluster, limit_date: date) -> list[Cluster]:
        """시계열 클러스터 탐색
        주어진 클러스터와 동일한 토픽을 지닌 과거 클러스터를 추출한다.

        :param target_cluster: 기준이 될 클러스터
        :param limit_date: DB에서 조회할 날짜 제한(금일부터 limit_date까지만을 조회)
        :return: 기준 클러스터의 시계열 클러스터
        """
        target_topic = self._preprocessed_cluster_repository.find_all_by_cluster(clusters=target_cluster)[0].words
        clusters: list[Cluster] = self._cluster_repository.find_all_by_section_id_and_limit_date(section_id=target_cluster.section_id,
                                                                                                 limit_date=limit_date)
        preprocessed_clusters: list[PreprocessedCluster] = self._preprocessed_cluster_repository.find_all_by_cluster(clusters=clusters)
        time_series_cluster = []

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
