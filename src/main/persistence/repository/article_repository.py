from datetime import datetime, date, timedelta
from typing import List, Union

from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import Article


class ArticleRepository(Repository):
    def __init__(self):
        super().__init__(Article)

    def find_all_by_section_id(self, section_id, t_date: date = None, duration: (datetime, datetime) = None):
        if t_date is not None:
            start = datetime.combine(t_date, datetime.min.time())
            end = start + timedelta(days=1)
            duration = (start, end)

        if duration is not None:
            query = And(Column('section_id', section_id), Between('regdate', *duration))
        else:
            query = Column('section_id', section_id)

        return self.find_all_by(query)

    def find_all_by_cluster_id(self, cluster_id: int) -> List[Article]:
        query = Column('cluster_id', cluster_id)
        return self.find_all_by(query)

    def count_by_cluster_id(self, cluster_id: int) -> int:
        query = Column('cluster_id', cluster_id)
        return self.count_by(query)

