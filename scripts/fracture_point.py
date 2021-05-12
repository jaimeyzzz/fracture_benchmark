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
inPos = []
frPos = []
N = 6
for j in range(-N, N + 1):
    for i in range(-N, N + 1):
        if j % 2 == 0:
            i = i + 0.5
        pos = []
        if math.sqrt(3.0) * i + j <= 0.0:
            pos = inPos
        elif math.sqrt(3) * i + j <= 1.7:
            pos = frPos
        pos.append((x + r * 2.0 * i, y + r * math.sqrt(3) * j))

for x, y in inPos:
    circle = plt.Circle((x, y), r * 0.95, color='#0088ff', fill=True)
    ax.add_patch(circle)
    # ax.plot([x, x + r], [y, y - dr], color='lightgray', linewidth=1)
    # ax.plot([x, x - r], [y, y - dr], color='lightgray', linewidth=1)
    # ax.plot([x, x - 2.0 * r], [y, y], color='lightgray', linewidth=1)
for x, y in frPos:
    circle = plt.Circle((x, y), r * 0.95, color='#00cc11', fill=True)
    ax.add_patch(circle)
    # ax.plot([x, x + r], [y, y - dr], color='lightgray', linewidth=1)
    # ax.plot([x, x - r], [y, y - dr], color='lightgray', linewidth=1)
    # ax.plot([x, x - 2.0 * r], [y, y], color='lightgray', linewidth=1)
    ax.plot([x, x + 0.5 * dr], [y, y + 0.5 * r], color='red', linewidth=3)

frX = [x + 0.5 * math.sqrt(3) * r for x, y in frPos]
frY = [y + 0.5 * r for x, y in frPos]
ax.plot(frX, frY, 'black', linestyle='dashed', marker='o', linewidth=3, markersize=6)

for x, y in frPos:
    frx = x + 0.5 * dr
    fry = y + 0.5 * r
    ax.arrow(frx, fry, 0.5 * dr, 0.5 * r, color='black', linewidth=3, head_width=0.01, head_length=0.01)

plt.savefig('fracture_point.pdf', bbox_inches='tight', pad_inches=0)
plt.show()