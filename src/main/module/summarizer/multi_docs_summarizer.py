from .rdass import RDASS
from rouge import Rouge
from dataclasses import dataclass
from .kobart_summarizer import KoBARTSummarizer
from sentence_transformers import SentenceTransformer
from persistence.models import Article, PreprocessedArticle


@dataclass
class News:
    title: str
    lead: str
    content: str
    summary: str


@dataclass
class Centroid:
    article: Article
    score: float
    summary: str


class MultiDocsSummarizer:
    def __init__(self, sentence_bert: SentenceTransformer):
        self._rouge = Rouge()
        self._rdass = RDASS(sentence_bert)
        self._summarizer = KoBARTSummarizer()

    def _get_rdass_score(self, news: News) -> float:
        """뉴스의 RDASS 점수 산출"""
        if news.summary == "":
            # 리드는 이미 정답인 요약문이므로 RDASS 점수를 산정하지 않음
            return 1
        else:
            return self._rdass.get_scores(docs=news.content, ref=news.title, predict=news.summary)

    def _get_rouge_score(self, news: News, topics: list[str]) -> float:
        """뉴스의 rouge 점수 산출"""
        if news.summary == "":
            return self._rouge.get_scores(hyps=news.lead, refs=' '.join(topics))[0]['rouge-1']['f']
        else:
            return (self._rouge.get_scores(hyps=news.summary, refs=news.content)[0]['rouge-1']['f'] +
                    self._rouge.get_scores(hyps=news.summary, refs=news.lead)[0]['rouge-1']['f'] +
                    self._rouge.get_scores(hyps=news.summary, refs=' '.join(topics))[0]['rouge-1']['f'])

    def _get_news_list(self,
                       article_list: list[Article],
                       preprocessed_list: list[PreprocessedArticle]) -> list[News]:
        """주어진 문서에서 뉴스 타입 리스트 반환"""
        news_list: list[News] = []

        for article, preprocessed_article in zip(article_list, preprocessed_list):
            news_list.append(News(title=article.title,
                                  lead=preprocessed_article.lead,
                                  content=article.content,
                                  summary=preprocessed_article.summary))

        return news_list

    def summarize(self,
                  article_list: list[Article],
                  preprocessed_list: list[PreprocessedArticle],
                  topics: list[str]) -> Centroid:
        """다중 문서 요약 수행"""
        news_list: list[News] = self._get_news_list(article_list=article_list,
                                                    preprocessed_list=preprocessed_list)
        centroid: Centroid = Centroid(Article(), 0, "")

        for idx, news in enumerate(news_list):
            cur_rdass: float = self._get_rdass_score(news=news)
            cur_rouge: float = self._get_rouge_score(news=news,
                                                     topics=topics)

            if centroid.score < cur_rdass + cur_rouge:
                centroid.article = article_list[idx]
                centroid.score = cur_rdass + cur_rouge
                centroid.summary = news.lead + " " + news.summary

        return centroid
