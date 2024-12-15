import read_camera
import numpy as np
import matplotlib.pyplot as plt

a = read_camera.read_camera()
b = np.frombuffer(a, dtype=np.uint16).reshape((1024,1536))
b = b[::-1]

plt.imshow(b, cmap="hot", vmin=np.min(b), vmax=np.max(b))
plt.axis('off')
plt.show()