import cProfile
import logging
from abc import ABC, abstractmethod
from pstats import Stats, SortKey
from threading import Thread
from time import sleep


def create_logger(
        logger_name: str,
        logfile_path: str,
        stream_level: int = logging.ERROR,
        file_level: int = logging.DEBUG,
) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(stream_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class LoggingThread(Thread):
    def __init__(self, thread_name: str, logfile_path: str):
        super().__init__()
        self.thread_name = thread_name
        self.logfile_path = logfile_path
        self.logger = create_logger(thread_name, logfile_path)


class TestLoggingThread(LoggingThread):
    def __init__(self, thread_name: str, logfile_path: str, sleep_time: float):
        super().__init__(thread_name, logfile_path)
        self.sleep_time = sleep_time

    def run(self):
        for i in range(100):
            self.logger.debug(f'{i}')
            sleep(self.sleep_time)


def test_multi_thread_logging():
    logger = create_logger('test_multi_thread_logging', 'test_multi_thread_logging.log')
    logger.info('test_multi_thread_logging')
    t1 = TestLoggingThread('t1', 'test_multi_thread_logging.log', 0.2)
    t2 = TestLoggingThread('t2', 'test_multi_thread_logging.log', 0.3)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


class ProfileLoggingThread(LoggingThread, ABC):
    def __init__(self, thread_name: str, logfile_path: str, profile_path: str):
        super().__init__(thread_name, logfile_path)
        self.profile_path = profile_path

    @abstractmethod
    def run_impl(self):
        pass

    def run(self):
        profile = cProfile.Profile()
        try:
            profile.runcall(self.run_impl)
        except Exception as e:
            self.logger.error(f'{e}')
        finally:
            profile.dump_stats(self.profile_path)


class TestProfileLoggingThread(ProfileLoggingThread):
    def __init__(self, thread_name: str, logfile_path: str, profile_name: str, sleep_time: float):
        super().__init__(thread_name, logfile_path, profile_name)
        self.sleep_time = sleep_time

    def run_impl(self):
        for i in range(100):
            self.logger.debug(f'{i}')
            sleep(self.sleep_time)


def test_multi_thread_profile():
    logger = create_logger('test_multi_thread_logging', 'test_multi_thread_logging.log')
    logger.info('test_multi_thread_logging')
    t1 = TestProfileLoggingThread('t1', 'test_multi_thread_logging.log', 'profile%d.profile', 0.2)
    t2 = TestProfileLoggingThread('t2', 'test_multi_thread_logging.log', 'profile%d.profile', 0.3)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def print_profile(profile_name: str):
    ps = Stats(profile_name)
    ps.strip_dirs().sort_stats(SortKey.TIME).print_stats(.3)


if __name__ == '__main__':
    print_profile('profile8004.profile')

__all__ = [
    'create_logger',
    'LoggingThread',
    'ProfileLoggingThread'
]