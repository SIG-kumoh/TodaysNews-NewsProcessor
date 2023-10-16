from typing import Union, List

from module.custom_orm import Repository
from module.custom_orm.query import *

from persistence.models import PreprocessedArticle, Article


class PreprocessedArticleRepository(Repository):
    def __init__(self):
        super().__init__(PreprocessedArticle)

    def find_all_by_article(self, articles: Union[Article, List[Article]]):
        if isinstance(articles, Article):
            articles = [articles]
        article_id_list = [article.article_id for article in articles]

        def executable(sess):
            res = sess.query(PreprocessedArticle).filter(PreprocessedArticle.article_id.in_(article_id_list)).all()
            return res

        return self.exec(executable)
