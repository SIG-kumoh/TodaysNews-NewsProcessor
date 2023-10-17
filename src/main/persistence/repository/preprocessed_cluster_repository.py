from typing import Union, List

from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import PreprocessedCluster, Cluster


class PreprocessedClusterRepository(Repository):
    def __init__(self):
        super().__init__(PreprocessedCluster)

    def find_all_by_cluster(self, clusters: Union[Cluster, List[Cluster]]):
        if isinstance(clusters, Cluster):
            clusters = [clusters]
        cluster_ids = [cluster.cluster_id for cluster in clusters]

        def executable(sess):
            res = sess.query(PreprocessedCluster).filter(PreprocessedCluster.cluster_id.in_(cluster_ids)).all()
            return res

        return self.exec(executable)
