import numpy as np
import matplotlib.pyplot as plt
import sys

# Create data
colors = (0,0,0)
area = np.pi*3*4*4*4

x = np.zeros([65536]) # max size of the pointset to load
y = np.zeros([65536])

f = open(str(sys.argv[1]), "r")

u = 0

for t in f:
        
    line = t.split()
    x[int(u)] = float(line[0])
    y[int(u)] = float(line[1])
    u += 1

print(u)
fig = plt.figure(figsize=(int(np.sqrt(u)/2), int(np.sqrt(u)/2)), dpi=80)
ax = fig.add_subplot(1, 1, 1)

plt.scatter(x[0:int(u/2)], y[0:int(u/2)], s=area, c=(1, 0, 0), alpha=1.0)
plt.scatter(x[int(u/2):u], y[int(u/2):u], s=area, c=(0, 0, 1), alpha=1.0)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.tight_layout()

plt.savefig(str(sys.argv[2]))
