import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_bbox(corners):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 12 edges of a cube connecting the corners
    edges = [
        [0,1,3,2], [4,5,7,6], [0,1,5,4],
        [2,3,7,6], [0,2,6,4], [1,3,7,5]
    ]
    for edge in edges:
        poly = Poly3DCollection([corners[edge]], alpha=0.25, facecolor='cyan')
        ax.add_collection3d(poly)

    ax.scatter(corners[:,0], corners[:,1], corners[:,2], color='r')
    plt.show()
