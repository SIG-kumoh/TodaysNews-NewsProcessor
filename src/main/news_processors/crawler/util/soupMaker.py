import ssl
from bs4 import BeautifulSoup
import urllib.request
from urllib.error import URLError, HTTPError
import requests
import random
from time import sleep


def open_url(url, logger=None, count=5, sleep_time=3):    # 오류에 대해 최대 5회 재접근
    sleep(random.randrange(0, sleep_time))
    result = None
    source = None

    if 0 < count:
        try:
            headers = {'User-Agent': 'Chrome/66.0.3359.181'}
            if url[4] == 's':
                context = ssl._create_unverified_context()
                req = urllib.request.Request(url, headers=headers)
                html = urllib.request.urlopen(req, context=context)
                source = html.read()
                html.close()
            else:
                source = requests.get(url, headers=headers).content
        except WindowsError as e:
            if logger is not None:
                logger.debug(f"[remain try : {str(count)}] win error : {url}")
            result = open_url(url, logger, count - 1, sleep_time)
        except HTTPError as e:
            if logger is not None:
                logger.debug(f"[remain try : {str(count)}] {str(e.getcode())} error : {url}")
            result = open_url(url, logger, count - 1, sleep_time)

        if source is not None:
            result = BeautifulSoup(source, "html5lib")

    return result