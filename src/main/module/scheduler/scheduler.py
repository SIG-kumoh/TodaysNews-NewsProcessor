import time
import schedule

from enum import Enum, auto
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Thread, Lock
from abc import ABC, abstractmethod


class Operation(Enum):
    START: int = auto


class Schedule(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Scheduler:
    def __init__(self):
        self.background_task_scheduler = None
        self.processes = {}
        self.schedules = []

    def add_schedule(self, name: str, schedule_class, period: str, term: int, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        c_pipe, p_pipe = Pipe(duplex=False)

        process = Process(target=_run_process, daemon=True, args=(c_pipe, schedule_class, *args), kwargs=kwargs)
        process.start()

        def trigger():
            p_pipe.send(Operation.START)

        cur_schedule = (schedule.every(term).__getattribute__(period), trigger)

        self.processes[name] = process
        self.schedules.append(cur_schedule)

    def start(self):
        self.background_task_scheduler = Thread(target=self._background_schedule, daemon=True)
        self.background_task_scheduler.start()

    def run_forever(self):
        self.background_task_scheduler.join()

    def _background_schedule(self):
        for cur_schedule, trigger in self.schedules:
            cur_schedule.do(trigger)

        while True:
            schedule.run_pending()
            time.sleep(1)


def _run_process(c_pipe: Connection, schedule_class: Schedule, *args, **kwargs):
    lock = Lock()
    target = schedule_class(*args, **kwargs)

    def start_schedule():
        with lock:
            target(*args, **kwargs)

    while True:
        opr = c_pipe.recv()
        if not lock.locked() and opr == Operation.START:
            t = Thread(target=start_schedule, daemon=True)
            t.start()
        else:
            print('이미 실행 중 입니다.')
