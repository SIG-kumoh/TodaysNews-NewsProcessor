from datetime import date, timedelta

from module.scheduler import Scheduler, Schedule
from news_processors import Crawler, ClusterMaker

# scheduler = Scheduler()

if __name__ == '__main__':
    # scheduler.add_schedule('test', Test1, period='seconds', term=10, args=('asdf',))
    # scheduler.start()
    # scheduler.run_forever()
    crawling_start = Crawler()
    clustering_start = ClusterMaker()

    start = date(year=2023, month=10, day=20)
    end = date(year=2023, month=11, day=3)

    t_date = start
    while t_date <= end:
        print(f'{t_date} start')
        crawling_start(t_date)
        clustering_start(t_date)
        t_date = t_date + timedelta(days=1)


