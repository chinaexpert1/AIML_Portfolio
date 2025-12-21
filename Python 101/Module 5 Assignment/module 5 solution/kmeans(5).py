import math
import matplotlib.pyplot as plt

# function to draw a scatterplot of centroids and clusters
# you can also put pieces of this function directly into the relevant spots in the code
# parameters: centroids (list), clusters (list)
def plot_cluster(centroids, clusters):
    colors = ["black", "blue", "green", "orange"]
    plt.xlabel("x-coordinates") # label x-axis
    plt.ylabel("y-coordinates") # label y-axis

    # collect all x-values, y-values for drawing with matplotlib
    # uses a list comprehension, could also use a regular for loop
    x = [point[0] for point in centroids] # collect x-values in a list
    y = [point[1] for point in centroids] # collect y-values in a list

    plt.scatter(x, y, color="red", s=100) # draw scatterplot to the canvas

    # collect all x-values, y-values for drawing with matplotlib
    for i, cluster in enumerate(clusters):
        x = [point[0] for point in cluster] # collect x-values in a list
        y = [point[1] for point in cluster] # collect y-values in a list

        plt.scatter(x, y, color=colors[i]) # draw scatterplot to the canvas

    # render canvas to the screen
    plt.show()

################################################################################

cluster_changes = 0
clusters = []
distances = []
patients = []
prev_cluster_size = []
X = 0 # used for convenient indexing of x-coordinates
Y = 1 # used for convenient indexing of y-coordinates

################################################################################

# open points.txt for reading points from file
f = open("points2.txt", "r")

iterations = int(f.readline()) # read number of clustering iterations from file
num_points = int(f.readline()) # read number of points from file
num_clusters = int(f.readline()) # read number of clusters from file
num_patients = num_points - num_clusters

################################################################################

# create a list of k initially empty clusters (lists)
# uses a list comprehension, could also use a regular for loop
clusters = [[] for i in range(num_clusters)]

# create a list of k previous cluster sizes, each initially 0 (empty)
# uses a list comprehension, could also use a regular for loop
prev_cluster_size = [0 for i in range(num_clusters)]

################################################################################

# create a list of centroids using
centroids = []

# read and parse centroids from file into a list
for line in range(num_clusters):
    point = f.readline().strip().split(',') # remove newlines, split on commas
    x = int(point[0]) # parse x-coordinate
    y = int(point[1]) # parse y-coordinate
    centroids.append([x, y]) # add point to list
print(centroids)
################################################################################
  
# read and parse points from file into a list
for line in range(num_patients):
    point = f.readline().strip().split(',') # remove newlines, split on commas
    x = int(point[0]) # parse x-coordinate
    y = int(point[1]) # parse y-coordinate
    patients.append([x, y]) # add point to list

################################################################################

print(f"Initial COVID-19 Patients: {centroids}\n")

for r in range(iterations):
    for point in patients:
        # calculate the distance between the current point and each of the 4 centroids
        for i in range(len(centroids)):
            distance = math.sqrt((centroids[i][X] - point[0]) ** 2 + (centroids[i][Y] - point[1]) ** 2)
            distances.append(distance)

        # assign each point to the cluster to which it is closest
        for i in range(len(distances)):
            if distances[i] == min(distances):
                clusters[i].append(point)

        # clear distances list before next iteration
        distances.clear()

    # check if any points have changed clusters
    for i in range(len(clusters)):
        if len(clusters[i]) != prev_cluster_size[i]:
            cluster_changes += 1
            break

    # calculate the resulting centroid for each cluster using the mean
    for i in range(len(clusters)):
        total_x = 0
        total_y = 0

        # sum the x and y coordinates of each point in the current cluster
        for point in clusters[i]:
            total_x += point[X]
            total_y += point[Y]

        # update centroid with the resulting cluster's mean x and y values
        centroids[i][X] = total_x / len(clusters[i])
        centroids[i][Y] = total_y / len(clusters[i])

        # store current cluster size for comparison in next iteration
        prev_cluster_size[i] = len(clusters[i])

    # draw a scatterplot of the initial clustering
    if r == 0:
        plot_cluster(centroids, clusters)

    # empty the cluster except on final iteration
    if r < iterations - 1:
        for cluster in clusters:
            cluster.clear()

# output number of iterations until clusters are stable
print(f"Iterations to achieve stability: {cluster_changes}\n")

print("Final Centroids:")
for centroid in centroids:
    print(f"{centroid}")

print()

# output the centroid, the number of points, and a list of points for each cluster
for i in range(len(clusters)):
    print(f"Number of patients in Cluster {i}: {len(clusters[i])}")
    print(f"{clusters[i]}\n")

# draw a scatterplot of the final clustering
plot_cluster(centroids, clusters)