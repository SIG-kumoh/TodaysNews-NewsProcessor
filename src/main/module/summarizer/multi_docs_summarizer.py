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
        """뉴스의 RDASS 점수 산출

        컨텐츠, 제목, 리드와 함께 후반부 요약문에 대한 RDASS 점수를 산출한다.
        만일 본문의 길이가 짧아 후반부 요약문이 뽑히지 않았다면 0점을 반환한다.
        """
        if news.summary == "":
            return 0
        else:
            return (self._rdass.get_scores(docs=news.content, ref=news.title, predict=news.summary) +
                    self._rdass.get_scores(docs=news.content, ref=news.lead, predict=news.summary)) / 2

    def _get_rouge_score(self, news: News) -> float:
        """뉴스의 rouge 점수 산출

        제목, 리드와 함께 후반부 요약문에 대한 ROUGE 점수를 산출한다.
        만일 본문의 길이가 짧아 후반부 요약문이 뽑히지 않았다면 0점을 반환한다.
        """
        try:
            if news.summary == "":
                return 0
            else:
                return (self._rouge.get_scores(hyps=news.summary, refs=news.title)[0]['rouge-1']['r'] +
                        self._rouge.get_scores(hyps=news.summary, refs=news.lead)[0]['rouge-1']['r']) / 2
        except:
            return 0

    def _get_relation_score(self, news: News, topics: list[str]) -> float:
        """뉴스의 연관 점수 산정
        
        제목, 리드, 컨텐츠, 후반부 요약문과 함께 토픽에 대한 ROUGE1 RECALL 점수를 산정하여
        해당 뉴스와 클러스터와의 관련도를 측정한다.
        """
        topic = ' '.join(topics)
        score = (self._rouge.get_scores(hyps=news.title, refs=topic)[0]['rouge-1']['r'] +
                 self._rouge.get_scores(hyps=news.content, refs=topic)[0]['rouge-1']['r'])
        
        try:
            if news.summary == "":
                if news.lead == news.content:
                    return score / 2
                else:
                    score += self._rouge.get_scores(hyps=news.lead, refs=topic)[0]['rouge-1']['r']
                    return score / 3
            else:
                score += self._rouge.get_scores(hyps=news.lead, refs=topic)[0]['rouge-1']['r']
                score += self._rouge.get_scores(hyps=news.summary, refs=topic)[0]['rouge-1']['r']
                return score / 4
            
        except:
            return 0

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
        centroid: Centroid = Centroid(Article(), -1, "")

        for idx, news in enumerate(news_list):
            cur_rdass: float = self._get_rdass_score(news=news)
            cur_rouge: float = self._get_rouge_score(news=news)
            cur_relation: float = self._get_relation_score(news=news, topics=topics)

            if centroid.score < cur_rdass + cur_rouge + cur_relation:
                centroid.article = article_list[idx]
                centroid.score = cur_rdass + cur_rouge + cur_relation
                centroid.summary = news.lead + " " + news.summary

        return centroid
