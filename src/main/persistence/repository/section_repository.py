from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import Section


class SectionRepository(Repository):
    def __init__(self):
        super().__init__(Section)
