import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img1 = mpimg.imread('test_images/test3.jpg')
img2 = mpimg.imread('test_images/test4.jpg')
img3 = mpimg.imread('test_images/test5.jpg')

fig = plt.figure()
"""
subplot(xyz)
x: kaç satır olacağı
y: kaç sütun olacağı
z: kaçıncı resim olacağını belirliyor...
"""

plt.subplot(211)
plt.imshow(img1)
plt.title('Car Positions Before HeatMap')
plt.subplot(223)
plt.imshow(img2)
plt.title('Car Positions After HeatMap')
plt.subplot(224)
plt.imshow(img3, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()

