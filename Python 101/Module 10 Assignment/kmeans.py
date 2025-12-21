import math

cluster_changes = 0
clusters = []
points = []
prev_cluster_size = []

# open points.txt for reading points from file
f = open("points.txt", "r")

# read number of clustering iterations from file
iterations = int(f.readline())
# read number of points from file
num_points = int(f.readline())
# read number of clusters from file
num_clusters = int(f.readline())
# read cluster indexes from file
cluster_indexes = []

for i in range(num_clusters):
  cluster_indexes.append(int(f.readline()))

# create a list of k initially empty clusters (lists)
clusters = [[] for i in range(num_clusters)]

# create a list of k previous cluster sizes, initially 0
prev_cluster_size = [0 for i in range(num_clusters)]

# read and parse points from file into a list
for line in range(num_points):
  point = f.readline().strip().split(",")
  try:
    x = int(point[0])
    y = int(point[1])
  except ValueError:
    continue
  except IndexError:
    continue
  points.append([x, y])

# create a list of centroids using
centroids = []
for index in range(len(cluster_indexes)):
  index_value = cluster_indexes[index]
  centroids.append(points[index_value])
  # use a generator expression to create a list instead
  # centroids = [points[index] for index in cluster_indexes]

# remove the centroids from the list of data points
for index in range(len(cluster_indexes)):
  index_value = cluster_indexes[index]
  del points[index_value]
  
print(f"Initial COVID-19 Patients: {centroids}\n")

for r in range(iterations):
  for point in points:
    # calculate the distance between the current point and each of the 4 centroids
    d1 = math.sqrt((centroids[0][0]-point[0])**2 + (centroids[0][1]-point[1])**2)
    d2 = math.sqrt((centroids[1][0]-point[0])**2 + (centroids[1][1]-point[1])**2)
    d3 = math.sqrt((centroids[2][0]-point[0])**2 + (centroids[2][1]-point[1])**2)
    d4 = math.sqrt((centroids[3][0]-point[0])**2 + (centroids[3][1]-point[1])**2)

    # assign point to the cluster with the nearest centroid
    if d1 == min(d1, d2, d3, d4):
      clusters[0].append(point)
    elif d2 == min(d1, d2, d3, d4):
      clusters[1].append(point)
    elif d3 == min(d1, d2, d3, d4):
      clusters[2].append(point)
    else:
      clusters[3].append(point)

  # check if any points have changed clusters
  for i in range(len(clusters)):
    if len(clusters[i]) != prev_cluster_size[i]:
      cluster_changes += 1
  else:
    break

  # calculate the resulting centroid for each cluster using the mean
  for i in range(len(clusters)):
    # sum the x and y coordinates of each point in the current cluster
    total_x = 0
    total_y = 0
    for point in clusters[i]:
      total_x += point[0]
      total_y += point[1]

    # update centroid with the resulting cluster's mean x and y values
    centroids[i][0] = total_x/len(clusters[i])
    centroids[i][1] = total_y/len(clusters[i])

    # store current cluster size for comparison in next iteration
    prev_cluster_size[i] = len(clusters[i])

  # empty the cluster except on final iteration
  if r < iterations:
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