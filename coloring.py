# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:19:41 2017

@author: marco
"""
import numpy as np
import matplotlib.colors 
from collections import deque
class Coloring:
    walls1 = np.array([
            # horizontal
            [[0, 500], [1000, 500]],
            [[400, 400], [500, 400]],
            [[600, 400], [700, 400]],
            [[800, 400], [1000, 400]],
            [[200, 300], [400, 300]],
            [[100, 200], [200, 200]],
            [[400, 200], [700, 200]],
            [[200, 100], [300, 100]],
            [[600, 100], [900, 100]],
            [[0, 0], [1000, 0]],
            # vertical
            [[0, 0], [0, 500]],
            [[100, 100], [100, 200]],
            [[100, 300], [100, 500]],
            [[200, 200], [200, 400]],
            [[200, 0], [200, 100]],
            [[300, 100], [300, 200]],
            [[300, 400], [300, 500]],
            [[400, 100], [400, 400]],
            [[500, 0], [500, 200]],
            [[600, 100], [600, 200]],
            [[700, 200], [700, 300]],
            [[800, 200], [800, 400]],
            [[900, 100], [900, 300]],
            [[1000, 0], [1000, 500]],
        ])
    walls2 = np.array([
            # horizontal
            [[0, 900], [1500, 900]],
            [[100, 800], [400, 800]],
            [[500, 800], [600, 800]],
            [[800, 800], [1000, 800]],
            [[1100, 800], [1200, 800]],
            [[1300, 800], [1400, 800]],
            [[100, 700], [600, 700]],
            [[700, 700], [800, 700]],
            [[1000, 700], [1100, 700]],
            [[1200, 700], [1400, 700]],
            [[900, 600], [1200, 600]],
            [[1300, 600], [1500, 600]],
            [[0, 500], [100, 500]],
            [[1300, 500], [1400, 500]],
            [[100, 400], [200, 400]],
            [[1200, 400], [1400, 400]],
            [[300, 300], [800, 300]],
            [[900, 300], [1200, 300]],
            [[400, 200], [600, 200]],
            [[700, 200], [800, 200]],
            [[1200, 200], [1500, 200]],
            [[200, 100], [300, 100]],
            [[500, 100], [700, 100]],
            [[800, 100], [900, 100]],
            [[1100, 100], [1400, 100]],
            [[0, 0], [1500, 0]],
            # vertical
            [[0, 0], [0, 900]],
            [[100, 0], [100, 300]],
            [[100, 500], [100, 600]],
            [[100, 700], [100, 800]],
            [[200, 100], [200, 200]],
            [[200, 300], [200, 400]],
            [[200, 500], [200, 700]],
            [[300, 100], [300, 300]],
            [[400, 0], [400, 200]],
            [[500, 800], [500, 900]],
            [[700, 100], [700, 200]],
            [[700, 700], [700, 800]],
            [[800, 200], [800, 800]],
            [[900, 100], [900, 700]],
            [[1000, 0], [1000, 200]],
            [[1000, 700], [1000, 800]],
            [[1100, 700], [1100, 800]],
            [[1100, 100], [1100, 300]],
            [[1200, 800], [1200, 900]],
            [[1200, 400], [1200, 700]],
            [[1300, 200], [1300, 300]],
            [[1300, 500], [1300, 600]],
            [[1400, 300], [1400, 500]],
            [[1400, 700], [1400, 800]],
            [[1500, 0], [1500, 900]],
        ])
    walls3 = np.array([
            # horizontal
            [[0, 1300], [2000, 1300]],
            [[100, 1200], [500, 1200]],
            [[600, 1200], [1400, 1200]],
            [[1600, 1200], [1700, 1200]],
            [[0, 1100], [600, 1100]],
            [[1500, 1100], [1600, 1100]],
            [[1600, 1000], [1800, 1000]],
            [[800, 1000], [900, 1000]],
            [[100, 1000], [200, 1000]],
            [[700, 900], [800, 900]],
            [[1600, 900], [1800, 900]],
            [[200, 800], [300, 800]],
            [[800, 800], [1200, 800]],
            [[1300, 800], [1500, 800]],
            [[1600, 800], [1900, 800]],
            [[900, 700], [1400, 700]],
            [[1500, 700], [1600, 700]],
            [[1700, 700], [1900, 700]],
            [[700, 600], [800, 600]],
            [[1400, 600], [1500, 600]],
            [[1600, 600], [1700, 600]],
            [[100, 500], [200, 500]],
            [[300, 500], [500, 500]],
            [[600, 500], [700, 500]],
            [[1400, 500], [1900, 500]],
            [[100, 400], [200, 400]],
            [[400, 400], [600, 400]],
            [[1500, 400], [1600, 400]],
            [[1700, 400], [1800, 400]],
            [[200, 300], [300, 300]],
            [[400, 300], [500, 300]],
            [[600, 300], [800, 300]],
            [[900, 300], [1100, 300]],
            [[1300, 300], [1500, 300]],
            [[1600, 300], [1700, 300]],
            [[100, 200], [200, 200]],
            [[500, 200], [600, 200]],
            [[800, 200], [1100, 200]],
            [[1200, 200], [1400, 200]],
            [[1500, 200], [1600, 200]],
            [[200, 100], [300, 100]],
            [[500, 100], [800, 100]],
            [[1000, 100], [1200, 100]],
            [[1400, 100], [1600, 100]],
            [[1800, 100], [1900, 100]],
            [[0, 0], [2000, 0]],
            # vertical
            [[0, 0], [0, 1300]],
            [[100, 0], [100, 300]],
            [[100, 400], [100, 1000]],
            [[200, 300], [200, 400]],
            [[200, 600], [200, 800]],
            [[200, 900], [200, 1000]],
            [[300, 100], [300, 600]],
            [[300, 800], [300, 1100]],
            [[400, 0], [400, 300]],
            [[400, 1200], [400, 1300]],
            [[500, 100], [500, 200]],
            [[600, 200], [600, 400]],
            [[600, 1100], [600, 1200]],
            [[700, 200], [700, 300]],
            [[700, 400], [700, 1100]],
            [[800, 100], [800, 200]],
            [[800, 300], [800, 500]],
            [[800, 600], [800, 700]],
            [[800, 1000], [800, 1100]],
            [[900, 0], [900, 100]],
            [[900, 300], [900, 600]],
            [[900, 900], [900, 1200]],
            [[1000, 100], [1000, 200]],
            [[1200, 100], [1200, 200]],
            [[1300, 0], [1300, 100]],
            [[1400, 100], [1400, 700]],
            [[1500, 700], [1500, 1000]],
            [[1500, 1100], [1500, 1200]],
            [[1600, 200], [1600, 400]],
            [[1600, 600], [1600, 700]],
            [[1600, 1000], [1600, 1100]],
            [[1600, 1200], [1600, 1300]],
            [[1700, 1100], [1700, 1200]],
            [[1700, 700], [1700, 800]],
            [[1700, 500], [1700, 600]],
            [[1700, 0], [1700, 300]],
            [[1800, 100], [1800, 400]],
            [[1800, 600], [1800, 700]],
            [[1800, 900], [1800, 1200]],
            [[1900, 800], [1900, 1300]],
            [[1900, 100], [1900, 600]],
            [[2000, 0], [2000, 1300]],
        ])
    def __init__(self, maze=1):
        
        if(maze==1):
            self.map_x=10
            self.map_y=5
            self.color_grid = np.zeros((10,5,3))
            self.walls = self.walls1
            self.breadth_first_coloring(0,4,0)
            self.breadth_first_coloring(9,4,1)
            self.breadth_first_coloring(6,1,2)
        elif(maze==2):
            self.map_x=15
            self.map_y=9
            self.color_grid = np.zeros((15,9,3))
            self.walls = self.walls2
            #self.breadth_first_coloring(0, 4, 0)
            self.breadth_first_coloring(1,7,0)
            #self.breadth_first_coloring(12, 3, 1)
            self.breadth_first_coloring(13,5,1)
            #self.breadth_first_coloring(8,5,2)
            self.breadth_first_coloring(13, 7, 2)
        elif(maze==3):
            self.map_x=20
            self.map_y=13
            self.color_grid = np.zeros((20,13,3))
            self.walls = self.walls3
            #self.breadth_first_coloring(2,5,0)
            #self.breadth_first_coloring(12,8,1)
            #self.breadth_first_coloring(18,6,2)
            self.breadth_first_coloring(2, 3, 0)
            self.breadth_first_coloring(14, 5, 1)
            self.breadth_first_coloring(17, 7, 2)
        else:
            print("Could not find the maze for coloring") 

    def breadth_first_coloring(self,start_x,start_y,color_i, for_topo = False):
        open_set = deque()
        closed_set = set()
        depth = dict()
        depth[(start_x,start_y)] = 0.0
        open_set.append((start_x,start_y))
        
        while len(open_set)>=1:
            parent = open_set.popleft()
            
            for child in self.get_neighbors(parent[0],parent[1]):
                if child in closed_set:
                    continue
                elif child not in open_set:
                    open_set.append(child)
                    depth[child]= depth[parent]+1.0
            closed_set.add(parent)
            
        max_depth = np.max(list(depth.values()))
        for x,y in depth.keys():
            if(for_topo):
                self.color_grid[x,y,color_i] = depth[(x,y)] 
            else:
                self.color_grid[x,y,color_i] = depth[(x,y)] /  max_depth
    
    def vectorize_topo_map(self):
        self.color_grid = np.zeros((self.map_x,self.map_y,self.map_x * self.map_y))
        for i in range(self.map_x):
            for j in range(self.map_y):
                self.breadth_first_coloring(i,j,j+i*self.map_y, for_topo = True)    
    
        self.topological_dist = np.reshape(self.color_grid,(self.map_x * self.map_y,self.map_x * self.map_y))
        return self.topological_dist
            
    def get_topo_distance(self, classified_indices, correct_indices, maze=1):
        self.vectorize_topo_map()
        
        return self.topological_dist[list(map(int,classified_indices)),list(map(int,correct_indices))]
    
    def get_topo_distance_by_pose(self,start,goal):
        startindex = int(start[1]/100)+int(start[0]/100)*self.map_y
        goalindex = int(goal[1]/100)+int(goal[0]/100)*self.map_y
        return self.topological_dist[startindex,goalindex]
        
    def get_topo_pose_index(self, poses):   
    
        indices = np.zeros(np.shape(poses)[0], dtype=np.int32)
        for i in range(np.shape(poses)[0]):
            x = int(poses[i,0]/100)
            y = int(poses[i,1]/100)
            indices[i] = y+x*self.map_y
        return indices
    
    def get_pose_by_topo(self, indices, mean = 0, std = 1):
        pose = np.zeros((np.shape(indices)[0], 3))
        for i in range(np.shape(indices)[0]):
            x = int(indices[i]/self.map_y)*100 + 50
            y = (indices[i]%self.map_y)*100 + 50
            pose[i,:] = [x + np.random.normal(0,15), y + np.random.normal(0,15), 0 ]
        return (pose - mean) / std
        
    def get_neighbors(self,x_raw,y_raw):
        x = x_raw*100 + 50
        y = y_raw*100 + 50
        neighbors = []        
        #Right:
        if not np.any([ w[0,0] == w[1,0] and w[1,0] == x+50 and w[0,1] <= y and y <= w[1,1] for w in self.walls]):
            neighbors.append((x_raw+1,y_raw))
        #Left
        if not np.any([ w[0,0] == w[1,0] and w[1,0] == x-50 and w[0,1] <= y and y <= w[1,1] for w in self.walls]):
            neighbors.append((x_raw-1,y_raw))
        #Up
        if not np.any([ w[0,1] == w[1,1] and w[1,1] == y+50 and w[0,0] <= x and x <= w[1,0] for w in self.walls]):
            neighbors.append((x_raw,y_raw+1))
        #Down
        if not np.any([ w[0,1] == w[1,1] and w[1,1] == y-50 and w[0,0] <= x and x <= w[1,0] for w in self.walls]):
            neighbors.append((x_raw,y_raw-1))
        
        return neighbors
    
    def get_color(self,poses):
        
        colors = np.zeros((np.shape(poses)[0],3))
        for i in range(np.shape(poses)[0]):
            x = int(poses[i,0]/100)
            y = int(poses[i,1]/100)
            colors[i,:] = self.color_grid[x,y,:]

        return colors
    
    def get_grid_color(self,poses):
        #min_x, min_y = np.min(poses,axis=0)
        max_x, max_y = np.max(poses,axis=0)/100
        max_x = int(max_x)+1
        max_y = int(max_y)+1
        colors = np.zeros(np.shape(poses)[0])
        for i in range(np.shape(poses)[0]):
            x = int(poses[i,0]/100)
            y = int(poses[i,1]/100)
            colors[i] = [float(x)/max_x,float(y)/max_y,float(x+y)/(max_x+max_y)]
        return colors   
    
    def get_sequential_color(self, observation):
        pass

    def get_orientational_color(self,orientations):
        min_a = np.abs(np.min(orientations))
        orientations +=min_a
        orientations /=  np.max(orientations)
        n = np.shape(orientations)[0]
        
        colors = np.ones((n,3))
        colors[:,0] = orientations
        colors = matplotlib.colors.hsv_to_rgb(colors)
        return colors
