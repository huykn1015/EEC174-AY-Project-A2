
import matplotlib.pyplot as plt
import numpy as np
import heapq
import argparse

parser = argparse.ArgumentParser(
                    prog='maze_solver',
                    description='solve mazes',
                    epilog='Text at the bottom of help')
parser.add_argument('maze_path', type=str, help='specify maze file')
parser.add_argument('--v', help='Visutal maze path finding', action='store_true')
parser.add_argument('--algorithm', type=str, help='specify algorithm used: dijkstra/astar')


def man_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def save_out(maze, start, stop, final):
    global maze_path
    maze_name = maze_path.split('.')[0]
    outfile = open(maze_name + '_out.txt', 'w')
    maze[start[0]][start[1]] = 's'
    maze[stop[0]][stop[1]] = 'e'
    for y,x in final:
        maze[x][y] = 'x'
    for row in maze:
        outfile.write(''.join(row) + '\n')
def read_maze(file_path):
    """
    Read the maze from a text file and convert it into a 2D numpy array.
    
    Parameters:
    - file_path: str, the path to the maze text file.
    
    Returns:
    - maze: 2D numpy array, representation of the maze where ' ' indicates a path, '#' indicates a wall,
            's' is the start, and 'e' is the end.
    """
    maze = []
    binary_maze = []
    open_spaces = []
    maze_file = open(file_path, 'r')
    prev = 0
    start = None
    end = None
    for i, line in enumerate(maze_file.readlines()):
        if 's' in line:
            start = (line.find('s'), i)
        if 'e' in line:
            end = (line.find('e'), i)
        maze.append([c for c in line][:-1])
        binary_maze.append(([(0 if c == '#' else 1) for c in line])[:-1])
    prev = len(binary_maze[0])
    if start == None:
        print("Maze is missing a starting space")
        exit(1)
    if end == None:
        print("Maze is missing an end space")
        exit(1)
    print(start, end)
    for i in range(len(binary_maze)):
        if len(binary_maze[i]) != prev:
            print('Invalid maze shape')
            exit(1)
        for j in range(len(binary_maze[0])):
            if maze[i][j] != '#':
                open_spaces.append((j, i))
    return (maze, binary_maze, start, end, open_spaces)


def a_star_algorithm(maze_):
    """
    Implement the A* algorithm to find the shortest path through the maze from start to end.
    
    Parameters:
    - maze: 2D numpy array, the maze to navigate.
    
    Add more parameters if you see fit
    """
    delta = ((1,0), (-1,0), (0, 1), (0, -1))
    maze, binary_maze, start, stop, open_spaces= maze_
    max_dist = len(maze) * len(maze[0])
    unvisited = [(0, start, start)] #space: dist
    visited = {} #space: [dist, parent]
    final = []
    found = False
    while unvisited and not found:
        dist, space, parent = heapq.heappop(unvisited)
        visited[space] = (dist, parent)
        plot_maze(maze_, current=space, visited=visited.keys())
        if space == stop:
            curr = visited[space]
            while curr[1] != start:
                final.append(curr[1])
                curr = visited[curr[1]]
            found = True
            save_out(maze, start, stop, final)
        for dx, dy in delta:
            next = (space[0] + dx, space[1] + dy)
            if next in visited or next not in open_spaces:
                continue
            newDist = man_dist(next, stop)
            heapq.heappush(unvisited, (newDist, next, space))
                
    if not found:
        print("Maze is not solveable")
    else: plot_maze(maze_, current=stop, visited=visited.keys(), final=final)
    global visualize
    if visualize: input('Press Any key to close program')


def dijkstras_algorithm(maze_):
    """
    Implement Dijkstra's algorithm to find the shortest path through the maze from start to end.
    
    Parameters:
    - maze: 2D numpy array, the maze to navigate.
    
    Add more parameters if you see fit
    """
    
    delta = ((1,0), (-1,0), (0, 1), (0, -1))
    maze, binary_maze, start, stop, open_spaces= maze_
    max_dist = len(maze) * len(maze[0])
    unvisited = [(0, start, start)] #space: dist
    visited = {} #space: [dist, parent]
    final = []
    found = False
    while unvisited and not found:
        dist, space, parent = heapq.heappop(unvisited)
        visited[space] = (dist, parent)
        plot_maze(maze_, current=space, visited=visited.keys())
        if space == stop:
            curr = visited[space]
            while curr[1] != start:
                final.append(curr[1])
                curr = visited[curr[1]]
            found = True
            save_out(maze, start, stop, final)
        for dx, dy in delta:
            next = (space[0] + dx, space[1] + dy)
            if next in visited or next not in open_spaces:
                continue
            newDist = dist + 1
            heapq.heappush(unvisited, (newDist, next, space))
    if not found:
        print("Maze is not solveable")
    else: 
        plot_maze(maze_, current=stop, visited=visited.keys(), final=final)
    global visualize
    if visualize: input('Press Any key to close program')


def plot_maze(maze_, current = None, visited = None, final = None):
    """
    Visualize the maze, the visited cells, the current position, and the end position.
    
    Parameters:
    - maze: 2D numpy array, the maze to visualize.
    - visited: visited coordinates.
    - current: current position in the maze.

    Add more parameters if you see fit
    """
    global visualize
    if not visualize:
        return
    maze, binary_maze, start, stop, open_spaces = maze_
    plt.imshow(binary_maze, cmap='gray')
    if visited and current != stop:
        for x,y in visited:
           if(x,y) == start or (x,y) == stop:
               continue
           plt.plot(x, y , 'yo') 
    elif visited and final and current == stop:
        for x,y in final:
            if(x,y) == start or (x,y) == stop:
               continue
            plt.plot(x, y , 'go') 
        for x,y in [(x,y) for x,y in visited if (x,y) not in final]:
            if(x,y) == start or (x,y) == stop:
               continue
            plt.plot(x, y , 'yo') 
    if current and current != stop and current != start:
        plt.plot(current[0], current[1], 'go')
    plt.plot(start[0], start[1], 'b^')
    plt.plot(stop[0], stop[1] , 'rv')
    plt.plot()
    plt.pause(.25)
    




"""
Parse command-line arguments and solve the maze using the selected algorithm.
"""

args = parser.parse_args()
visualize = args.v
maze_path = args.maze_path
alg = args.algorithm
maze = read_maze(maze_path)
if args.algorithm == 'dijkstra':
    dijkstras_algorithm(maze)
elif args.algorithm == 'astar':
    a_star_algorithm(maze)
else:
    #print(args.algorithm)
    print('%s is not a valid algorthm. Please specify either "dijkstra" or "astar"'%(args.algorithm))
    exit(1)