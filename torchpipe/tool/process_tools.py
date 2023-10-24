# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
import psutil


class Resource:
    def __init__(self, pid):
        self.p = psutil.Process(pid)

    def get_cpu_mem(self):
        return self.p.cpu_percent(), self.p.memory_percent()


if __name__ == "__main__":
    # get pid from args
    if len(sys.argv) < 2:
        print("missing pid arg")
        sys.exit()

    # get process
    pid = int(sys.argv[1])
    p = psutil.Process(pid)

    # monitor process and write data to file
    interval = 3  # polling seconds
    with open("process_monitor_" + p.name() + '_' + str(pid) + ".csv", "a+") as f:
        f.write("time,cpu%,mem%\n")  # titles
        while True:
            current_time = time.strftime(
                '%Y%m%d-%H%M%S', time.localtime(time.time()))
            # better set interval second to calculate like:  p.cpu_percent(interval=0.5)
            cpu_percent = p.cpu_percent()
            mem_percent = p.memory_percent()
            line = current_time + ',' + \
                str(cpu_percent) + ',' + str(mem_percent)
            print(line)
            f.write(line + "\n")
            time.sleep(interval)
