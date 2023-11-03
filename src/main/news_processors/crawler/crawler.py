import logging

import yaml
from konlpy.tag import Okt

from sentence_transformers import SentenceTransformer

from datetime import datetime, date

from module.scheduler import Schedule
from module.summarizer.kobart_summarizer import KoBARTSummarizer
from persistence.models import PreprocessedArticle, Article, Section
from persistence.repository import PreprocessedArticleRepository, ArticleRepository, SectionRepository

from .util import soupMaker, remove_tag, simply_ws, only_BMP_area, LimitLoader
from .custom_tokenizer import CustomTokenizer


CONFIG_PATH = 'resources/config/crawler/config.yml'


class Crawler(Schedule):
    def __init__(self):
        self.conf = {}
        self.load_config()

        self.section_id = {}

        self.embedding_model = SentenceTransformer(self.conf['EMBEDDING_MODEL'])
        self.tokenizer = CustomTokenizer(Okt())
        self.summarizer = KoBARTSummarizer()

        # TODO 리팩터링
        self.logger = logging.getLogger('crawler')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __call__(self, t_date: date = None):
        if t_date is None:
            t_date = date.today()
        self.load_config()
        self.load_section()

        article_list, preprocessed_list = self.crawling(t_date)
        if article_list:
            article_repository = ArticleRepository()
            preprocessed_article_repository = PreprocessedArticleRepository()

            article_repository.insert(article_list)
            article_iter = iter(article_list)
            for preprocessed_article in preprocessed_list:
                preprocessed_article.article_id = article_iter.__next__().article_id
            preprocessed_article_repository.insert(preprocessed_list)

        self.logger.info("crawling finished")

    def load_config(self):
        with open(CONFIG_PATH, 'r', encoding='UTF-8') as yml:
            self.conf = yaml.safe_load(yml)

    def load_section(self):
        section_repository = SectionRepository()
        section_list = section_repository.find_all()
        for section in section_list:
            self.section_id[section.section_name] = section.section_id

    def crawling(self, t_date: date) -> (list, list):
        self.logger.info("crawling start")

        article_list = []
        preprocessed_list = []
        for press in self.conf['CODE']:
            code = self.conf['CODE'][press]

            self.logger.debug(str(press) + ' crawling start..')
            news_url_list = self.get_updated_url_list(code, t_date)
            cur_article_list = self.get_news_list(news_url_list, press)
            cur_preprocessed_list = self.preprocess(cur_article_list)

            article_list += cur_article_list
            preprocessed_list += cur_preprocessed_list

        return article_list, preprocessed_list

    def get_updated_url_list(self, code, t_date: date) -> list:
        limit = LimitLoader.get_limit()
        result = []

        loop = True
        req_page = 1
        while loop:
            page_url = self._make_page_url(code, t_date, req_page)
            page_soup = soupMaker.open_url(url=page_url, logger=self.logger)
            res_page = remove_tag(page_soup.find('div', class_='paging').find('strong'))

            if res_page == str(req_page):
                url_list = self._find_article_url(page_soup)

                for cur_url in url_list:
                    if cur_url > limit[code]:
                        result.append(cur_url)
                    else:
                        loop = False

                req_page += 1
            else:
                loop = False

        result = sorted(list(set(result)), reverse=True)

        self.logger.debug('find ' + str(len(result)) + ' news')
        if len(result) > 0:
            self.logger.debug('Limit.yml - ' + code + ' update')
            limit[code] = result[0]
            LimitLoader.update_limit(limit)

        return result

    def get_news_list(self, url_list: list, press: str) -> list[Article]:
        article_list = []

        for url in url_list:
            article = self.get_article(url, press)
            if article is not None:
                article_list.append(article)

        return article_list

    def get_article(self, url, press) -> Article:
        soup = soupMaker.open_url(url, logger=self.logger)

        article = None
        try:
            title = only_BMP_area(remove_tag(soup.find('h2', id='title_area')))[:150]
            date_str = soup.find('span', class_='media_end_head_info_datestamp_time')['data-date-time']
            regdate = datetime(year=int(date_str[0:4]), month=int(date_str[5:7]), day=int(date_str[8:10]),
                               hour=int(date_str[11:13]), minute=int(date_str[14:16]), second=int(date_str[17:19]))
            img_url = soup.find('img', id='img1')['data-src']
            soup.find('span', class_='end_photo_org').decompose()
            content_soup = soup.find('div', id='newsct_article')
            strong_list = content_soup.find_all('strong')
            if len(strong_list) > 0:
                strong_list[0].decompose()
            content = only_BMP_area(simply_ws(remove_tag(content_soup)))
            if len(content) < 50:
                raise Exception
            writer = only_BMP_area(remove_tag(soup.find('span', class_='byline_s')))[:100]
            section_name = remove_tag(soup.find('em', class_='media_end_categorize_item'))

            article = Article(
                regdate=regdate,
                img_url=img_url,
                url=url,
                press=press,
                title=title,
                content=content,
                writer=writer,
                section_id=self.section_id[section_name],
            )
        except:
            pass

        return article

    def preprocess(self, article_list) -> list[PreprocessedArticle]:
        self.logger.debug('preprocessing..')

        preprocessed_list = []
        for article in article_list:
            if len(article.content) > 650:
                summary = self.summarizer.summarize(article.content[len(article.content) // 2:])
            else:
                summary = ""

            preprocessed_list.append(PreprocessedArticle(
                tokens=self.tokenizer(article.__getattribute__(self.conf['TOKENIZING_TARGET'])),
                embedding=self.embedding_model.encode(article.__getattribute__(self.conf['EMBEDDING_TARGET']),
                                                      show_progress_bar=False),
                summary=summary))
        return preprocessed_list

    def _find_article_url(self, soup) -> [str]:
        url_list = []
        targets = ['type06_headline', 'type06']

        for tg in targets:
            ul_tag = soup.find('ul', class_=tg)
            if ul_tag is not None:
                url_list += [attr['href'] for attr in ul_tag.select('a')]

        return url_list

    def _make_page_url(self, code: str, t_date: date, page: int) -> str:
        url = self.conf['HOME'] + code + \
              self.conf['DATE'] + t_date.strftime('%Y%m%d') + \
              self.conf['PAGE'] + str(page)
        return url
