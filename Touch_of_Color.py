from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.image as mpimg
from PIL import Image
import cv2
import itertools
from PIL import Image
import time

def labcontrast(img):
    ##########################################
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = final
    return img
    ##########################################
# FIRST NFT 20ETH

fig = plt.figure(frameon=False)
img = cv2.imread("Photos/Womens/W3.jpg")
h, w, c = img.shape
# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert to grayscale
img = labcontrast(img)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# create a binary thresholded image

th = 110
_, binary = cv2.threshold(gray, th, th, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# fig.set_size_inches(w,h)
# fig.set_size_inches(w, h, forward=True)
with_contours = cv2.drawContours(binary, contours, -1, (0, 255, 0), 1)
contour = list(map(np.squeeze, contours))

ims = []
delta_x = 500
delta_y = 550
xdata, ydata = [], []
x = []
y = []
random.seed(2021)
for i in range(1, len(contour) - 1):
    cnt = len(contour[i]) - 1
    for j in range(0, cnt):
        contour_list = contour[i][j].tolist()
        if type(contour_list) in (list, tuple, dict, str):
            xcnt, ycnt = contour_list
            x.append(xcnt)
            y.append(ycnt)
        else:
            pass
            # print("not a list")

x_new = []
y_new = []
max_iter = 100
frames = []
figure = fig
for m in range(0, max_iter):
    plt.axis("off")
    plt.imshow(with_contours, cmap='gray')
    fig.subplots_adjust(right=1, top=1, left=0, bottom=0, wspace=0, hspace=0)
    plt.margins(0, 0)
    rand_idx = int(random.random() * len(x))
    x_new.append(x[rand_idx])
    y_new.append(y[rand_idx])
    # x_new = np.linspace(x[0], x[-1], 50)
    # y_new = f(x_new)
    # plt.plot(x,y,'-.', x_new, y_new)
    colors = np.random.rand(len(x_new))
    area = (50 * np.random.rand(len(x_new))) ** 2  # 0 to 15 point radii
    # imgplot = plt.imshow(imgAdpt)
    im, = plt.plot(x_new, y_new)
    #im = plt.scatter(x_new, y_new, s=area, c=colors, alpha=0.5)
    ims.append([im])
    #plt.savefig('Photos/ProducedNFT/Touch_of_Color/{:03}.png'.format(m), bbox_inches='tight', pad_inches=0, dpi=400)
    #new_frame = Image.open('Photos/temp/{:03}.png'.format(m))
    #frames.append(new_frame)
    if m > 5:
        del x_new[1]
        del y_new[1]
    else:
        pass
    #plt.clf()

ani = ArtistAnimation(figure, ims, interval=100, repeat_delay=100,
                    blit=True)

# plt.savefig('Photos/temp/test1.png', bbox_inches='tight',pad_inches = 0,dpi = 1000)
# ani.save('Photos/temp/test1.gif', writer='imagemagick', fps=12, )

#frames[0].save('Photos/ProducedNFT/Touch_of_Color/Touch_of_Color.gif', format='GIF', append_images=frames[1:],
#               save_all=True, duration=150, loop=0)
plt.show()
