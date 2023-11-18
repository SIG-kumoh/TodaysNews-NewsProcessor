from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# TODO USE CONFIG FILE
class EngineFactory:
    __username = 'root'
    __password = 'admin'
    __host = '202.31.202.34'
    __port = 443
    __db_name = 'todays_news'
    __engine = None

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__engine is None:
            self.__engine = create_engine(f'mysql+mysqlconnector://{self.__username}:{self.__password}@{self.__host}:{self.__port}/{self.__db_name}')

    def get_engine(self):
        return self.__engine

    def get_session(self):
        db_session = sessionmaker(self.__engine)
        return db_session()
