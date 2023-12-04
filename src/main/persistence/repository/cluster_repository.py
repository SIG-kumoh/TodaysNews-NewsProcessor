from datetime import datetime, date, timedelta
from typing import Union, List

from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import Cluster


class ClusterRepository(Repository):
    def __init__(self):
        super().__init__(Cluster)

    def find_all_by_duration(self, duration: Union[date, tuple]) -> List[Cluster]:
        if isinstance(duration, date):
            start = datetime.combine(duration, datetime.min.time())
            end = start + timedelta(days=1)
            duration = (start, end)

        query = Between('regdate', *duration)
        return self.find_all_by(query)

    def find_all_by_section_id(self,
                               section_id: int,
                               duration: Union[date, tuple] = None) -> List[Cluster]:
        if isinstance(duration, date):
            start = datetime.combine(duration, datetime.min.time())
            end = start + timedelta(days=1)
            duration = (start, end)

        query = And(Column('section_id', section_id),
                    Between('regdate', *duration))
        return self.find_all_by(query)

    def find_all_by_section_id_and_duration(self,
                                            section_id: int,
                                            duration: Union[date, tuple]) -> List[Cluster]:
        if isinstance(duration, date):
            start = datetime.combine(duration, datetime.min.time())
            end = start + timedelta(days=1)
            duration = (start, end)

        query = And(Column('section_id', section_id),
                    Between('regdate', *duration))
        return self.find_all_by(query)
