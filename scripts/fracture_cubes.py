import math
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('ggplot')

fig, ax = plt.subplots()
ax.set_aspect(1)
plt.axis('off')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
# FRACTURE POINTS

r = 0.08
x = 0.5
y = 0.5
dr = math.sqrt(3) * r
bPos = []
N = 6
for j in range(-N, N + 1):
    for i in range(-N, N + 1):
        if j % 2 == 0:
            i = i + 0.5
        pos = []
        if math.sqrt(3.0) * i + j > 1.7:
            pos = bPos
        elif math.sqrt(3.0) * i + j < -1.7:
            pos = bPos
        elif math.sqrt(3.0) * i - j > 5.7:
            pos = bPos
        pos.append((x + r * 2.0 * i, y + r * math.sqrt(3) * j))

for x, y in bPos:
    circle = plt.Circle((x, y), r * 0.95, color='#0088ff', fill=True)
    ax.add_patch(circle)
    if math.sqrt((x-0.4)**2+(y-0.6)**2) < 0.28:
        if math.sqrt(3.0) * x + y < 1.3:
            ax.plot([x, x + r], [y, y - dr], color='blue', linewidth=3)
            ax.plot([x, x - r], [y, y - dr], color='blue', linewidth=3)
            ax.plot([x, x - r], [y, y + dr], color='blue', linewidth=3)
            ax.plot([x, x - 2.0 * r], [y, y], color='blue', linewidth=3)
        else:
            ax.plot([x, x + r], [y, y + dr], color='blue', linewidth=3)
            ax.plot([x, x + r], [y, y - dr], color='blue', linewidth=3)
            ax.plot([x, x - r], [y, y + dr], color='blue', linewidth=3)
            ax.plot([x, x + 2.0 * r], [y, y], color='blue', linewidth=3)

circle = plt.Circle((0.4, 0.6), 0.3, color='black', fill=False, linestyle='dashed', linewidth=3)
ax.add_patch(circle)
circle = plt.Circle((0.4, 0.6), 0.01, color='black', fill=True)
ax.add_patch(circle)

plt.savefig('fracture_cubes.pdf', bbox_inches='tight', pad_inches=0)
plt.show()