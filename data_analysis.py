import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

data = np.load('output/peridynamics2.npz')
t = data['time']
f = data['load']

plt.plot(t, f)

plt.xlabel('$\delta$ / cm')
plt.ylabel('$F$ / N')

plt.show()