from typing import Union, List
from sqlalchemy import MetaData, select, func

from .query import *
from .engineFactory import EngineFactory


class Repository:
    def __init__(self, model_class):
        self.ef = EngineFactory()
        self.engine = self.ef.get_engine()
        self.meta_data = MetaData()
        self._model_class = model_class
        self.table_name = model_class.__tablename__

    def exec(self, executable):
        res = None
        with self.ef.get_session() as sess:
            try:
                res = executable(sess)
                sess.flush()
                sess.expunge_all()
                sess.commit()
            except Exception as e:
                print(e)
                sess.rollback()
        return res

    # CREATE
    def insert(self, model):
        if not isinstance(model, List):
            model = [model]

        def executable(sess):
            sess.add_all(model)

        self.exec(executable)

    # UPDATE
    def update(self, model):
        if not isinstance(model, List):
            model = [model]

        def executable(sess):
            res = []
            with sess.no_autoflush:
                for m in model:
                    m = sess.merge(m)
                    res.append(m)
            return res

        return self.exec(executable)

    # DELETE
    def delete(self, model):
        if not isinstance(model, List):
            model = [model]

        def executable(sess):
            with sess.no_autoflush:
                for m in model:
                    m = sess.merge(m)
                    sess.delete(m)

        self.exec(executable)

    # READ
    def find_by(self, query: Query):
        table = self._model_class.__table__
        query = select(table).where(query(table))

        def executable(sess):
            return sess.query(self._model_class).from_statement(query).one()
        return self.exec(executable)

    def find_all(self) -> List:
        table = self._model_class.__table__
        query = select(table)

        def executable(sess):
            return sess.query(self._model_class).from_statement(query).all()
        return self.exec(executable)

    def find_all_by(self, query: Query) -> List:
        table = self._model_class.__table__
        query = select(table).where(query(table))

        def executable(sess):
            return sess.query(self._model_class).from_statement(query).all()
        return self.exec(executable)

    def count_by(self, query: Query) -> int:
        table = Table(self.table_name, self.meta_data, autoload=True, autoload_with=self.engine)
        query = select([func.count()]).where(query(table))
        res = self.engine.execute(query).scalar()
        return res
