from .rdass import RDASS
from rouge import Rouge
from typing import Union
from itertools import zip_longest
from dataclasses import dataclass
from ._kobart_summarizer import KoBARTSummarizer
from sentence_transformers import SentenceTransformer


@dataclass
class News:
    title: str
    lead: str
    content: str
    summary: str


class MultiDocsSummarizer:
    def __init__(self, sentence_bert: SentenceTransformer):
        self._rouge = Rouge()
        self._rdass = RDASS(sentence_bert)
        self._summarizer = KoBARTSummarizer()

    def _get_rdass_score(self, news: News, regard_lead_as_summary: bool) -> float:
        """뉴스의 RDASS 점수 산출"""
        if regard_lead_as_summary:
            # 리드는 이미 정답인 요약문이므로 RDASS 점수를 산정하지 않음
            return 1
        else:
            return self._rdass.get_scores(docs=news.content, ref=news.title, predict=news.summary)

    def _get_rouge_score(self, news: News, topics: list[str], regard_lead_as_summary: bool) -> float:
        """뉴스의 rouge 점수 산출"""
        if regard_lead_as_summary:
            return self._rouge.get_scores(hyps=news.lead, refs=' '.join(topics))[0]['rouge-1']['f']
        else:
            return (self._rouge.get_scores(hyps=news.summary, refs=news.content)[0]['rouge-1']['f'] +
                    self._rouge.get_scores(hyps=news.summary, refs=news.lead)[0]['rouge-1']['f'] +
                    self._rouge.get_scores(hyps=news.summary, refs=' '.join(topics))[0]['rouge-1']['f'])

    def _get_news_list(self,
                       title_list: list[str],
                       content_list: list[str],
                       summary_list: list[str]) -> list[News]:
        """주어진 문서에서 뉴스 타입 리스트 반환"""
        news_list: list[News] = []

        for title, content, summary in zip_longest(title_list, content_list, summary_list, fillvalue=""):
            news_list.append(News(title=title, lead=content.split('.')[0], content=content, summary=summary))

        return news_list

    def summarize(self,
                  title_list: list[str],
                  content_list: list[str],
                  summary_list: list[str],
                  topics: list[str],
                  regard_lead_as_summary: bool = False) -> str:
        """다중 문서 요약 수행"""
        news_list: list[News] = self._get_news_list(title_list=title_list,
                                                    content_list=content_list,
                                                    summary_list=summary_list)
        representative_news: dict[str, Union[str, float]] = {'summary': "", 'score': 0}

        for news in news_list:
            cur_rdass: float = self._get_rdass_score(news=news,
                                                     regard_lead_as_summary=regard_lead_as_summary)
            cur_rouge: float = self._get_rouge_score(news=news,
                                                     topics=topics,
                                                     regard_lead_as_summary=regard_lead_as_summary)

            if representative_news['score'] < cur_rdass + cur_rouge:
                representative_news['summary'] = news.lead + '\n' + news.summary
                representative_news['score'] = cur_rdass + cur_rouge

        return representative_news['summary']
