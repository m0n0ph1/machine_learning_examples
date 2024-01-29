# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import matplotlib.pyplot as plt
import numpy as np

logits = np.log([ 0.1, 0.2, 0.3, 0.4 ])

samples = [ ]

for _ in range(10000):
    noise = np.random.random(len(logits))
    sample = np.argmax(logits - np.log(-np.log(noise)))
    samples.append(sample)

plt.hist(samples)
plt.show()
