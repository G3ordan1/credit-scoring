"""
A miner trapped in a mine containing 3 doors. The first door leads to a tunnel that takes
him to safety after 2 hours of travel. The second door leads to a tunnel that returns him to
the mine after 3 hours of travel. The third door leads to a tunnel that returns him to the
mine after 5 hours of travel.
Assuming that the miner is at all times equally likely to choose any one of the doors, what
is the expected length of time until the miner reaches safety.
"""
import numpy as np

p = [1/3, 1/3, 1/3]
t = [2, 3, 5]
iterations = 100000
total_times = []
for i in range(iterations):
    time = 0
    while True:
        door = np.random.choice(range(3), p=p)
        time += t[door]
        if door == 0:
            break
    total_times.append(time)

print(np.mean(total_times)) # 9.97 ~ 10

import matplotlib.pyplot as plt
plt.hist(total_times)
plt.show()