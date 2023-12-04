from module.scheduler import Scheduler
from news_processors import Crawler, ClusterMaker


if __name__ == '__main__':
    scheduler = Scheduler()
    scheduler.add_schedule('crawler', Crawler, period='minutes', term=10)
    scheduler.add_schedule('clusterMaker', ClusterMaker, period='hours', term=1)
    scheduler.start()
    scheduler.run_forever()
