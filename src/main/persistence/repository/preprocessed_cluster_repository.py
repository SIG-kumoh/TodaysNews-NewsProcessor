from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import PreprocessedCluster


class PreprocessedClusterRepository(Repository):
    def __init__(self):
        super().__init__(PreprocessedCluster)
