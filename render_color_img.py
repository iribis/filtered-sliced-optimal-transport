import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv


# Create data
colors = (0,0,0)
area = int(np.pi*3*4*3*3*2)

x = np.zeros([65536])
y = np.zeros([65536])

f = open(str(sys.argv[1]), "r")
u = 0

for t in f:
         
    line = t.split()

    x[int(u)] = float(line[0])
    y[int(u)] = float(line[1])
    u += 1

print(u)

fig = plt.figure(figsize=(int(np.sqrt(u)*1.5/2), int(np.sqrt(u)/2)), dpi=80)
ax = fig.add_subplot(1, 1, 1)

plt.scatter(x[0:int(u*float(sys.argv[3]))], y[0:int(u*float(sys.argv[3]))], s=area, c=(0,1,1), alpha=1.0)
plt.scatter(x[int(u*float(sys.argv[3])):int(u*float(sys.argv[4]))], y[int(u*float(sys.argv[3])):int(u*float(sys.argv[4]))], s=area, c=(1.0,0.0,1.0), alpha=1.0)
plt.scatter(x[int(u*float(sys.argv[4])):int(u*float(sys.argv[5]))], y[int(u*float(sys.argv[4])):int(u*float(sys.argv[5]))], s=area, c=(1.0,1.0,0.0), alpha=1.0)
plt.scatter(x[int(u*float(sys.argv[5])):int(u*float(sys.argv[6]))], y[int(u*float(sys.argv[5])):int(u*float(sys.argv[6]))], s=area, c=(0.0,0.0,0.0), alpha=1.0)

#ax.set_xlim([0.0, 0.7])
#ax.set_ylim([0.0, 1.0])

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.axis('off')
fig.tight_layout()

plt.savefig(str(sys.argv[2]))
