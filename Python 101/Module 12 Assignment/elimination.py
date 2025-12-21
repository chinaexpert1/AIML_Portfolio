'''Module 12 Assignment by Andrew Taylor atayl136'''
import networkx as nx
import pandas as pd
import math

# user input to switch input files
datafile = input("Please input data file with extension: ")

# read in file data to a variable and a list
with open(datafile) as reader:               # remember to switch to datafile variable
    numteams = reader.readline()
    dflist = [line.strip().split() for line in reader]

# generate some column labels for variable tables
columnheaders = ""
for i in range(len(dflist[0])):
    colstr = ''
    colstr += 'Col'
    colstr += str(i)
    colstr += ','
    columnheaders += (str(colstr))
columnheaders = columnheaders.rstrip(columnheaders[-1])

# create the list of team names
teamlist = ""
for row in range(len(dflist)):
    teamname = dflist[row][0]
    teamlist += teamname
    teamlist += ","
teamlist = teamlist.rstrip(teamlist[-1])
teamlist = teamlist.split(",")

# create iterable of games node names, and games teams edges with inf weight
gamesnodenames = ""
gamesteamsedgelist = []
inf = math.inf
for row in range(len(dflist)):
    for team2 in range(row+1, len(dflist)):
        teams = ""
        teams += dflist[row][0]
        teams += "-"
        teams += dflist[team2][0]
        gamesteamsedgelist.append((teams, dflist[row][0], inf))
        gamesteamsedgelist.append((teams, dflist[team2][0], inf))
        teams += ","
        gamesnodenames += teams
gamesnodenames = gamesnodenames.rstrip(gamesnodenames[-1])
gamesnodenames = gamesnodenames.split(",")

# create list of remaining games in same order as nodes
remaining = ""
for row in range(len(dflist)):
    for team2 in range(5+row, len(dflist[row])):
        games = ""
        games += dflist[row][team2]
        games += ","
        remaining += games
remaining = remaining.rstrip(remaining[-1])
remaining = remaining.split(",")
for number in range(len(remaining)):
    remaining[number] = int(remaining[number])

# turn remaining list into an edge list
remaininggamesedges = []
for e in range(len(gamesnodenames)):
    remaininggamesedges.append(('s', gamesnodenames[e], remaining[e]))

# convert numbers to int for trivial calculation and df creation
for line in range(len(dflist)):
    for entry in range(1, len(dflist[line])):
        dflist[line][entry] = int(dflist[line][entry])
            
# create pandas dataframe
df = pd.DataFrame(data = dflist, columns = columnheaders.split(','))

# scenario 1 elimination - more current than possible wins
trivial = []
for row in range(len(dflist)):
    if dflist[row][1] + dflist[row][3] < df['Col1'].max():
        trivial.append(True)         # True for each team eliminated
    else:                            # teams are treated in order
        trivial.append(False)        # now we have a list, respectively
        
# create a directed graph with weights for each team
g = nx.empty_graph(0, nx.DiGraph())
        
# scenario 2 elimination - max flow < saturation number
scenario2elimlist = []
for team in teamlist:
    g.clear()    
    graphteamlist = []                   # temporary lists for the graph
    graphnodenameslist = []
    graphgamesteamslist =[]
    remaininglist = []
    teamstotedgelist = []
    teamxwinsplusremaining = 0
    # populate temp lists excluding team x
    for i in range(len(teamlist)):
        if team == teamlist[i]:
            teamxwinsplusremaining += (df['Col1'].loc[i] + df['Col3'].loc[i])
    for i in range(len(remaininggamesedges)):
         if team not in remaininggamesedges[i][1]:
             remaininglist.append(remaininggamesedges[i])
    for i in range(len(teamlist)):
        if team != teamlist[i]:
            graphteamlist.append(teamlist[i])      
    for i in range(len(gamesnodenames)):
        if team not in gamesnodenames[i]:
            graphnodenameslist.append(gamesnodenames[i])
    for i in range(len(gamesteamsedgelist)):
        if team not in gamesteamsedgelist[i][0]:
            graphgamesteamslist.append(gamesteamsedgelist[i])
    for i in range(len(teamlist)):
        if team != teamlist[i]:
            teamstotedgelist.append((teamlist[i], 't', (teamxwinsplusremaining - df['Col1'].loc[i])))
    # start building graph      
    # add bucket nodes
    g.add_node('s')
    g.add_node('t')        
    # add team nodes
    g.add_nodes_from(graphteamlist)
    # add games nodes
    g.add_nodes_from(graphnodenameslist)
    # add weighted edges from s to games
    g.add_weighted_edges_from(remaininglist)
    # add weighted edges from games to teams
    g.add_weighted_edges_from(graphgamesteamslist)
    # add weighted edges from teams to t
    g.add_weighted_edges_from(teamstotedgelist)
    # calculate max flow and storw in a variable
    maxflow, flowdict = nx.maximum_flow(g, _s='s', _t='t', capacity='weight')
    # calculate saturation number
    saturationlist = []
    for i in range(len(remaininglist)):
        saturationlist.append(remaininglist[i][2])
    saturationnumber = sum(saturationlist)
    # add to list wether eliminated or not
    if maxflow < saturationnumber:
        scenario2elimlist.append(True)
    else:
        scenario2elimlist.append(False)
    # end loop
    
# elimination and print logic - checks lists
for team in range(len(teamlist)):
    if trivial[team] == True:
        print(f'{teamlist[team]} has been trivially eliminated.')
    elif scenario2elimlist[team] == True:
        print(f'{teamlist[team]} is eliminated.')
    else:
        print(f'{teamlist[team]} is not eliminated.')
        
        
# Bloated code, but it works!


