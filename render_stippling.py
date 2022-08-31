import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2 as cv
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

x = np.zeros([32*32*4*4*4*4])
y = np.zeros([32*32*4*4*4*4])
f = open(str(sys.argv[1]), "r")

area = int(np.pi*3*4*3*3*2)/10

count = 0
for t in f:
    line = t.split()
    x[int(count)] = float(line[0])
    y[int(count)] = float(line[1])
    count += 1

fig = plt.figure(figsize=(int(np.sqrt(count)/1.5/4), int(np.sqrt(count))/4), dpi=80)
ax = fig.add_subplot(1, 1, 1)

plt.scatter(x[:count], y[:count],s=area, c=(0,0,0))

#ax.set_xlim([0.0, 1.0])
#ax.set_ylim([0.0, 1.0])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.tight_layout()

plt.savefig(str(sys.argv[2]),bbox_inches='tight',pad_inches = 0)
