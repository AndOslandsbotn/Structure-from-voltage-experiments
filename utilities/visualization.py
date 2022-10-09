from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def find_convexHull(landmarks):
    return ConvexHull(landmarks)

def plotting_points_in_convexHull(landmarks, points_in_hull, points_not_in_hull, hull):
    # We plot the mds embedding in 2D so we must remove higher dimensions
    landmarks = landmarks[:, 0:2]
    points_in_hull = points_in_hull[:, 0:2]
    points_not_in_hull = points_not_in_hull[:, 0:2]

    for simplex in hull.simplices:
        plt.plot(landmarks[simplex, 0], landmarks[simplex, 1])

    plt.scatter(*landmarks.T, alpha=.5, color='k', s=200, marker='v')
    plt.scatter(points_in_hull[:, 0], points_in_hull[:, 1], marker='x', color='g')
    plt.scatter(points_not_in_hull[:, 0], points_not_in_hull[:, 1], marker='d', color='m')
    plt.show()

