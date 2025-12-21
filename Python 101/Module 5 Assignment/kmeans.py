
# K Means algorithm, by Andrew Taylor
# open data file
file_object = open("points1.txt", "r")

# max iterations
maxiter = file_object.readline()

# number of patients
N = file_object.readline()

# of clusters
K = file_object.readline()

# load patient points
patient_list = file_object.readlines()

# data loaded, time to close file
file_object.close()

# convert to int
maxiter = int(maxiter)
N = int(N)
K = int(K)
patientpairsnum = int((N-K))
numpairs = int(N/2)
pairssize = int(N/numpairs)

# strip newline and convert to int, took me hours couldn't find better way
patient_list = [item.strip() for item in patient_list]
patientliststring = str(patient_list)
patientliststring = patientliststring.strip("[")
patientliststring = patientliststring.strip("]")
patientliststring = patientliststring.replace("'", "")
patientliststring = patientliststring.replace("', '", ",")
patientliststring = patientliststring.replace(" ", "")
patientsall = [int(i) for i in patientliststring.split(",")]

#create list of list, empty groups, and append according to size
# set up remaining data structures
clusters = []
for i in range(K):
    clusters.append([])
resetclusters = clusters


# print(clusters)
    
centroids = []
for i in range(K):
    centroids.append([])
    
for i in range(K):
    for j in range(i * pairssize, (i+1) * pairssize):
        centroids[i].append(patientsall[j])
        
        
        
# print(centroids)

pairs = []
for i in range(N):
    pairs.append([])


for i in range(N):
    for j in range(i * 2, (i+1) * 2):
        pairs[i].append(patientsall[j])

pairs = pairs[K:]

distx = []
for i in range(K):
    distx.append([])

disty = []
for i in range(K):
    disty.append([])

dx = []
for i in range(K):
    dx.append([])

dy = []
for i in range(K):
    disty.append([])
    
distlist = []
#for i in range(K):
    #distlist.append([])
  
# cluster size
clustersize = []

#print(clustersize)

# dummy variable for comparing cluster size each iteration
dummyclusters = []

    
# initial iteration count and counter
currentiter = 0

#loop for iterations
for currentiter in range(maxiter):
    # loop to calculate distance and add to correct cluster
    for j in range(len(pairs)):
        for i in range(K):   
            distx[i] = abs(pairs[j][0] - centroids[i][0]) 
            disty[i] = abs(pairs[j][1] - centroids[i][1])
            eucliddist = ((distx[i]**2+disty[i]**2)**0.5)
            distlist.append(eucliddist)
        # print(distlist)
        closestclusterindex = distlist.index(min(distlist))
        clusters[closestclusterindex].append(pairs[j])
        distlist.clear()
    #print(centroids)
    #print(clusters)     
    # now check if patients changed clusters
    for i in range(K):   
        dummyclusters.append(len(clusters[i]))
    if dummyclusters == clustersize:
        break
    else:
        clustersize = dummyclusters
        dummyclusters.clear()
        currentiter += 1
    for i in range(K):  
        for j in range(len(clusters[i])):
            xtotal = 0
            xtotal += clusters[i][j][0]
            xmean = xtotal/len(clusters[i])
            ytotal = 0
            ytotal += clusters[i][j][1]
            ymean = ytotal/len(clusters[i])
        centroids[i] = xmean, ymean
        clusters[i].clear()

print(centroids)
#print(clusters)        


#print(clusters)


#print(clustersize)
#print(clusters)
# update centroid here
         # update clusters up here


#print(distlist)
#print(closestclusterindex)









