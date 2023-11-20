from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import RelatedCluster


class RelatedClusterRepository(Repository):
    def __init__(self):
        super().__init__(RelatedCluster)

