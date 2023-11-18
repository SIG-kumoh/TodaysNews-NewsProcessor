from datetime import date, timedelta

from module.scheduler import Scheduler, Schedule
from news_processors import Crawler, ClusterMaker


if __name__ == '__main__':
    # scheduler.add_schedule('test', Test1, period='minutes', term=10, args=('asdf',))

    # scheduler = Scheduler()
    # scheduler.add_schedule('crawler', Crawler, period='minutes', term=10)
    # scheduler.add_schedule('clusterMaker', ClusterMaker, period='hours', term=1)
    # scheduler.start()
    # scheduler.run_forever()

    # crawling_start = Crawler()
    clustering_start = ClusterMaker()
    clustering_start(date(year=2023, month=11, day=10))

    # start = date(year=2023, month=11, day=3)
    # end = date(year=2023, month=11, day=7)
    #
    # t_date = start
    # while t_date <= end:
    #     print(f'{t_date} start')
    #     crawling_start(t_date)
    #     clustering_start(t_date)
    #     t_date = t_date + timedelta(days=1)

# for cur in range(len(topic_words) - 1):
#     c = 0
#     for idx in range(len(labels)):
#         if labels[idx] == cur:
#             print(f'{idx} : {article_list[idx].title}')
#             c += 1
#     print(f'클러스터 크기 : {c}')
#     if c != 0:
#         topic_list = []
#         for topic in topic_words[cur]:
#             topic_list.append(topic[0])
#         print(f'토픽 : {topic_list}')
#     print()