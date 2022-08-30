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

# add specified number of subdivisions
nb_div = int(sys.argv[3])
color_grad = range(0,nb_div)
color_grad = [(a/nb_div,0,1.0-a/nb_div) for a in color_grad]

for subiv in range(0,nb_div):
    plt.scatter(x[int((subiv)*u/nb_div):int((subiv+1)*u/nb_div)], y[int((subiv)*u/nb_div):int((subiv+1)*u/nb_div)], s=area, c=color_grad[subiv], alpha=1.0)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.tight_layout()

plt.savefig(str(sys.argv[2]))
