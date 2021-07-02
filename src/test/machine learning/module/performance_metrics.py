"""
Title: Performance Metrics
Date: 2/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
Description: Calculate performance statistics of the object detector
"""

from time import perf_counter  # obtain time at highest precision possible


class Stat:
    def __init__(self):  # instantiate Stat class object
        self.latency = 0.0
        self.period = 0.0
        self.frame_count = 0

    def combine(self, new):
        """find total time and frame for average fps and latency calculation"""
        self.latency += new.latency
        self.period += new.period
        self.frame_count += new.frame_count


class PerformanceMetrics:
    def __init__(self, time_window=1.0):
        # 'time_window' defines the length of the timespan over which the 'current fps' value is calculated
        self.time_window_size = time_window
        self.last_stat = Stat()
        self.current_stat = Stat()
        self.total_stat = Stat()
        self.last_update_time = None

    def update(self, last_req_start_time, frame):
        # configurations for data overlay on frame
        current_time = perf_counter()  # timestamp

        if self.last_update_time is None:
            self.last_update_time = current_time

        self.current_stat.latency += current_time - last_req_start_time
        self.current_stat.period = current_time - self.last_update_time
        self.current_stat.frame_count += 1

        if current_time - self.last_update_time > self.time_window_size:
            self.last_stat = self.current_stat  # 'save' current stat
            self.total_stat.combine(
                self.last_stat)  # instantiate combine()
            self.current_stat = Stat()  # reset current_stat

            self.last_update_time = current_time

        current_latency, current_fps = self.get_last()
        return (current_latency, current_fps)

    def get_last(self):
        return (self.last_stat.latency / self.last_stat.frame_count
                if self.last_stat.frame_count != 0
                else None,
                self.last_stat.frame_count / self.last_stat.period
                if self.last_stat.period != 0.0
                else None)

    def get_total(self):
        frame_count = self.total_stat.frame_count + \
            self.current_stat.frame_count
        return (((self.total_stat.latency + self.current_stat.latency) / frame_count)
                if frame_count != 0
                else None,
                (frame_count / (self.total_stat.period +
                                self.current_stat.period))
                if frame_count != 0
                else None)

    def print_total(self):
        total_latency, total_fps = self.get_total()
        print("Latency: {:.1f} ms".format(total_latency * 1e3)
              if total_latency is not None else "Latency: N/A")
        print("FPS: {:.1f}".format(total_fps)
              if total_fps is not None else "FPS: N/A")
