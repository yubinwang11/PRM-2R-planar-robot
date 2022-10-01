from collections import defaultdict
import sys
import math
from turtle import Turtle, shape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse

from .Dijkstra import Graph, dijkstra, to_array
from .Utils import Utils

import random

def get_line_magnitude(x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        return lineMagnitude

def point_to_line_distance(point, start_point, end_point):
    px, py = point[0], point[1]
    x1, y1 = start_point
    x2, y2 = end_point

    line_magnitude = get_line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            # 点到直线的投影不在线段内, 计算点到两个端点距离的最小值即为"点到线段最小距离"
            ia = get_line_magnitude(px, py, x1, y1)
            ib = get_line_magnitude(px, py, x2, y2)
            if ia > ib:
                distance = ib
            else:
                distance = ia
        else:
            # 投影点在线段内部, 计算方式同点到直线距离, u 为投影点距离x1在x1x2上的比例, 以此计算出投影点的坐标
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            #point_mindist = np.array(ix, iy)
    
            distance = get_line_magnitude(px, py, ix, iy)
            
        return distance
        
class PRMController:
    def __init__(self, numOfRandomCoordinates, allObs, current, destination):
        self.numOfCoords = numOfRandomCoordinates
        self.coordsList = np.array([])
        self.allObs = allObs
        self.current = np.array(current)
        self.destination = np.array(destination)
        self.graph = Graph()
        self.utils = Utils()
        self.solutionFound = False
        
        self. circle_obs_center = np.array([65,65])
        self.circle_obs_radius = 10

    def runPRM(self, initialRandomSeed, saveImage=True):
        seed = initialRandomSeed
        # Keep resampling if no solution found
        while(not self.solutionFound):
            print("Trying with random seed {}".format(seed))
            np.random.seed(seed)

            # Generate n random samples called milestones
            self.genCoords()

            # Check if milestones are collision free
            self.checkIfCollisonFree()

            # Link each milestone to k nearest neighbours.
            # Retain collision free links as local paths.
            self.findNearestNeighbour()

            # Search for shortest path from start to end node - Using Dijksta's shortest path alg
            self.shortestPath()

            seed = np.random.randint(1, 100000)
            self.coordsList = np.array([])
            self.graph = Graph()

        if(saveImage):
            #plt.savefig("{}_samples.png".format(self.numOfCoords))
            plt.savefig('PRM 2R planar robot.eps',dpi=600,format='eps')
        plt.show()

    def genCoords(self, maxSizeOfMap=100):
        '''self.coordsList = np.random.randint(
            maxSizeOfMap, size=(self.numOfCoords, 2))
        #print(type(self.coordsList))
        #print(shape(self.coordsList.shape()))
        print(self.coordsList)
        '''
        self.coordsList = list()
        for i  in range(self.numOfCoords):
            l = 50 * math.sqrt(random.random())
            deg = random.random()  * math.pi *  2
            x = l * math.cos(deg) + 50
            y = l * math.sin(deg) + 50
            self.coordsList.append([x,y])
        
        self.coordsList = np.array(self.coordsList)
        
        # Adding begin and end points
        self.current = self.current.reshape(1, 2)
        self.destination = self.destination.reshape(1, 2)
        self.coordsList = np.concatenate(
            (self.coordsList, self.current, self.destination), axis=0)

    def checkIfCollisonFree(self):
        collision = False
        self.collisionFreePoints = np.array([])
        for point in self.coordsList:
            collision = self.checkPointCollision(point)
            if(not collision):
                if(self.collisionFreePoints.size == 0):
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point])
        self.plotPoints(self.collisionFreePoints)

    def findNearestNeighbour(self, k=5):
        X = self.collisionFreePoints
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 2), int)

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                if(not self.checkPointCollision(start_line) and not self.checkPointCollision(end_line)):
                    if(not self.checkLineCollision(start_line, end_line)):
                        self.collisionFreePaths = np.concatenate(
                            (self.collisionFreePaths, p.reshape(1, 2), neighbour.reshape(1, 2)), axis=0)

                        a = str(self.findNodeIndex(p))
                        b = str(self.findNodeIndex(neighbour))
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j+1])
                        x = [p[0], neighbour[0]]
                        y = [p[1], neighbour[1]]
                        plt.plot(x, y)

    def shortestPath(self):
        self.startNode = str(self.findNodeIndex(self.current))
        self.endNode = str(self.findNodeIndex(self.destination))

        dist, prev = dijkstra(self.graph, self.startNode)

        pathToEnd = to_array(prev, self.endNode)

        if(len(pathToEnd) > 1):
            self.solutionFound = True
        else:
            return

        # Plotting shorest path
        pointsToDisplay = [(self.findPointsFromNode(path))
                           for path in pathToEnd]

        x = [int(item[0]) for item in pointsToDisplay]
        y = [int(item[1]) for item in pointsToDisplay]
        plt.plot(x, y, c="blue", linewidth=3.5)

        pointsToEnd = [str(self.findPointsFromNode(path))
                       for path in pathToEnd]
        print("****Output****")

        print("The quickest path from {} to {} is: \n {} \n with a distance of {}".format(
            self.collisionFreePoints[int(self.startNode)],
            self.collisionFreePoints[int(self.endNode)],
            " \n ".join(pointsToEnd),
            str(dist[self.endNode])
        )
        )

        
    def checkLineCollision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.allObs:
            if(self.utils.isWall(obs)):
                uniqueCords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(
                    uniqueCords)
                if(line.intersection(wall)):
                    collision = True
            else:
                obstacleShape = shapely.geometry.Polygon(
                    obs.allCords)
                collision = line.intersects(obstacleShape)
                
            if(collision):
                return True
        
        if collision ==False:
            dist_obs = point_to_line_distance(self.circle_obs_center, start_line, end_line) 
            if dist_obs < self.circle_obs_radius:
                return True
            
        return False

    def findNodeIndex(self, p):
        return np.where((self.collisionFreePoints == p).all(axis=1))[0][0]

    def findPointsFromNode(self, n):
        return self.collisionFreePoints[int(n)]

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y, c="black", s=1)

    def checkCollision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        #if(obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.bottomLeft[1] <= p_y <= obs.topLeft[1]):
            #return True
        
        dist_point2obs = get_line_magnitude(p_x, p_y, obs[0], obs[1])
        if dist_point2obs < self.circle_obs_radius:
            return True
            
        return False

    def checkPointCollision(self, point):
        #for obs in self.allObs:
        collision = self.checkCollision(self.circle_obs_center, point)
        if(collision):
            return True
        return False
