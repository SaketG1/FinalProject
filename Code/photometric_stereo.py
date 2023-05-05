import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
import scipy
import skimage

from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

def Read(id, objectName, path = ""):
    source = plt.imread(path + objectName +  "/Image_" + id + ".png")
    source = skimage.transform.resize(source, (source.shape[0]/3, source.shape[1]/3))
    return source

def showDepthMap(depthmap):
    x, y = np.meshgrid(np.arange(0, depthmap.shape[1]), np.arange(0, depthmap.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ls = LightSource()
    color_shade = ls.shade(-depthmap, plt.cm.gray)
    ax.plot_surface(x, y, -depthmap, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis("off")
    plt.show()

def getDepth(N, shp, objectname):
    mask = plt.imread("../Images/" + objectname + "/mask.jpg")
    mask = skimage.transform.resize(mask, (mask.shape[0]/3, mask.shape[1]/3))
    plt.imshow(mask)
    plt.show()
    dict = {}
    count = 0
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            if mask[i,j,1] != 0:
                dict[(i, j)] = count
                dict[count] = (i, j)
                count += 1

    A = np.zeros((2*count, count))
    b = np.zeros(2*count)

    for i in range(count):
        row, col = dict[i]
        if (row, col + 1) in dict:
            A[(i*2), dict[(row, col)]] += 1
            A[(i*2), dict[(row, col + 1)]] += -1
            b[(i*2)] = N[row, col, 0] / N[row, col, 2]
        elif (row, col - 1) in dict:
            A[(i*2), dict[(row, col)]] += 1
            A[(i*2), dict[(row, col - 1)]] += -1
            b[(i*2)] = -1 * N[row, col, 0] / N[row, col, 2]
        if (row - 1, col) in dict:
            A[(i*2)+1, dict[(row, col)]] += 1
            A[(i*2)+1, dict[(row - 1, col)]] += -1
            b[(i*2)+1] = N[row, col, 1] / N[row, col, 2]
        elif (row + 1, col) in dict:
            A[(i*2)+1, dict[(row, col)]] += 1
            A[(i*2)+1, dict[(row + 1, col)]] += -1
            b[(i*2)+1] = -1 * N[row, col, 1] / N[row, col, 2]

    spA = scipy.sparse.lil_matrix(A)
    z = scipy.sparse.linalg.spsolve(spA.T @ spA, spA.T @ b)

    zmin = np.min(z)
    depthmap = np.zeros(shp)
    for i in range(z.shape[0]):
        row, col = dict[i]
        depthmap[row, col] = (z[i] - zmin)

    return depthmap


if __name__ == '__main__':
    inputDir = '../Images/'
    outputDir = '../Results/'
    objectName = 'cat'

    images = []
    lightDirs = []
    for index in range(1,20):
        source = Read(str(index+1).zfill(2),objectName, inputDir)
        images.append(source)

    with open(inputDir + objectName + "/light_directions.txt", 'r') as file:
        for line in file.readlines():
            d = []
            for num in line.split(" "):
                if num != '' and num != "\n":
                    d.append(float(num))
            lightDirs.append(d)
    Lt = np.asarray(lightDirs)

    shp = images[0].shape
    shwImg = np.zeros((images[0].shape[0], images[0].shape[1], 3))
    mask = np.zeros(shp)
    for i in range(shp[0]):
        for j in range(shp[1]):
            if images[0][i,j] < 0.03:
                shwImg[i,j] = [1, 0, 0]
                mask[i,j] = 0
            else:
                shwImg[i,j] = [0,1,0]
                mask[i,j] = 1

    I = np.zeros((Lt.shape[0], shp[0] * shp[1]))
    for i in range(len(images)):
        I[i,:] = images[i].flatten()

    L = np.transpose(Lt)

    G = np.matmul(np.matmul(inv(np.matmul(L,Lt)),L),I)
    albedo = np.zeros(G.shape[1])
    N = np.zeros(G.shape)
    for i in range(G.shape[1]):
        albedo[i] = np.linalg.norm(G[:, i])
        N[:, i] = G[:, i]/albedo[i]

    n1 = N[0, :].reshape(shp)
    n2 = N[1, :].reshape(shp)
    n3 = N[2, :].reshape(shp)
    N = np.stack([n1, n2, n3], axis=2)


    plt.imshow(N)
    plt.show()
    depthmap = getDepth(N, shp, objectName)

    # depthmap = np.zeros(shp)
    # for row in range(1, depthmap.shape[0]):
    #     if mask[row, 0] == 1:
    #         depthmap[row, 0] = depthmap[row-1, 0] + (n2[row, 0]/n3[row, 0])
    #
    # for row in range(depthmap.shape[0]):
    #     for col in range(1,depthmap.shape[1]):
    #         if mask[row, col] == 1:
    #             depthmap[row, col] = depthmap[row, col-1] + (n1[row, col]/n3[row, col])


    showDepthMap(depthmap)
