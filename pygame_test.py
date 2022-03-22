import pygame
from pygame.locals import *
import numpy as np
import random as rd

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

def carve_maze(grid:np.ndarray, size:int) -> np.ndarray:
    output_grid = np.empty([size*3, size*3],dtype=int)
    output_grid[:] = 0
    
    i = 0
    j = 0
    
    
    while i < size:
        previous_l = []
        w = i*3 + 1
        while j < size:
            k = j*3 + 1
            toss = grid[i,j]
            output_grid[w,k] = -1
            if toss == 0 and k+2 < size*3:
                output_grid[w,k+1] = -1
                output_grid[w,k+2] = -1
                previous_l.append(j)
            if toss == 1:
                # it's impossible to carve outside after preprocessing
                # look back, choose a random cell
                if grid[i,j-1] == 0:
                    # reaching from 0
                    # mandatory to be sure that previous_l has at least one element
                    # if we are coming from a list of previous cells, choose one and...
                    r = rd.choice(previous_l)
                    k = r * 3 + 1
                   
                # ...just carve north
                # this just carve north if this is the first element of the row (1 element loop)
                output_grid[w-1,k] = -1
                output_grid[w-2,k] = -1
                previous_l = []
            
            j += 1
            
        i += 1
        j = 0
        
    return output_grid

def preprocess_grid(grid:np.ndarray, size:int) -> np.ndarray:
    # fix first row and last column to avoid digging outside the maze external borders
    first_row = grid[0]
    first_row[first_row == 1] = 0
    grid[0] = first_row
    for i in range(1,size):
        grid[i,size-1] = 1
    return grid

def main(val):
    n=1
    p=0.5
    size=val

    # 1 (head) N, 0 (tail) E
    # np.random.seed(42)
    grid = np.random.binomial(n,p, size=(size,size))
    # print(grid)

    processed_grid = preprocess_grid(grid, size)
    # print(processed_grid)

    output = carve_maze(processed_grid, size)
    # print(output)

    return output

class Grid:
    def __init__(self,row_length, col_length):
        self.matrix = main(row_length)

    def setIndex(self, row, col, value):
        self.matrix[row, col] = value

    def getIndex(self, row, col):
        return self.matrix[row, col]

    def getRowLen(self):
        return self.matrix.shape[0]-1
    
    def getColLen(self):
        return self.matrix.shape[1]-1

    def setBound(self, val):
        for i in range(self.matrix.shape[1]):
            self.setIndex(0, i, -2);
            self.setIndex(self.getRowLen(), i, val)
    
        for i in range(self.matrix.shape[0]):
            self.setIndex(i, 0, -2);
            self.setIndex(i, self.getColLen(), val)

class Maze:
   def __init__(self, screen_w, screen_h):

    # screen dimensions
    self.screen_w = screen_w
    self.screen_h = screen_h

    # cell dimensions
    self.cell_width = 20
    self.cell_height = 20
    self.cell_margin = 5

    # init game and set screen dims
    pygame.init()

    # matrix for cells
    self.grid = Grid(
        self.screen_w // (self.cell_width+self.cell_margin), 
        self.screen_w // (self.cell_height+self.cell_margin) 
        )
    self.grid.setBound(-2)


    self.screen_w = screen_w * 3 - 10
    self.screen_h = screen_h * 3 - 10
    self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))

    # setup variables
    self.screen.fill(WHITE)
    self.background = BLACK
    self.running = True
    self.clock = pygame.time.Clock()
    self.drag=False
    self.clear = False

   def run(self):
        while self.running:
            for event in pygame.event.get():

                # Exit window
                if event.type == pygame.QUIT:
                    self.running = False
                
                # Background color keys
                elif event.type == KEYDOWN:
                    if event.key == K_g:
                        self.background = GREEN
                    elif event.key == K_b:
                        self.background = BLUE
                    elif event.key == K_r:
                        self.background = RED
                    elif event.key == K_w:
                        self.background = WHITE
                    elif event.key == K_k:
                        self.background = BLACK

                # Unclick stops events
                elif event.type == MOUSEBUTTONUP:
                    self.drag=False
                    self.clear=False
                
                # Erase grid colors
                elif self.drag and self.clear:
                    x_pos, y_pos = pygame.mouse.get_pos()

                    x_pos = x_pos // (self.cell_width+self.cell_margin)
                    y_pos = y_pos // (self.cell_height+self.cell_margin)

                    # if self.grid.getIndex(y_pos, x_pos) == 1:
                    self.grid.setIndex(y_pos, x_pos, 0)
                
                # Color grid cells
                elif self.drag:
                    x_pos, y_pos = pygame.mouse.get_pos()

                    x_pos = x_pos // (self.cell_width+self.cell_margin)
                    y_pos = y_pos // (self.cell_height+self.cell_margin)

                    if self.grid.getIndex(y_pos, x_pos) == 0:
                        self.grid.setIndex(y_pos, x_pos, 1)
                
                # Color single grid cell
                elif event.type == MOUSEBUTTONDOWN:
                    x_pos, y_pos = pygame.mouse.get_pos()

                    x_pos = x_pos // (self.cell_width+self.cell_margin)
                    y_pos = y_pos // (self.cell_height+self.cell_margin)

                    if self.grid.getIndex(y_pos, x_pos):
                        if self.grid.getIndex(y_pos, x_pos) == -1:
                            pass
                        else:
                            self.grid.setIndex(y_pos, x_pos, 0)
                        self.clear=True
                    else:
                        self.grid.setIndex(y_pos, x_pos, 1)

                    self.drag = True

            # fill background color
            self.screen.fill(self.background)

            # create grid
            for i in range(self.screen_h // (self.cell_height+self.cell_margin)):
                for j in range(self.screen_w // (self.cell_width+self.cell_margin)):
                    y_pos = i * (self.cell_height+self.cell_margin) +self.cell_margin
                    x_pos = j * (self.cell_width+self.cell_margin) + self.cell_margin

            # Black if cell is 1 white otherwise
                    if self.grid.getIndex(i, j) == -1:
                        pygame.draw.rect(self.screen, BLACK, (x_pos,y_pos, self.cell_width, self.cell_height))
                    elif self.grid.getIndex(i, j) == 0:
                        pygame.draw.rect(self.screen, WHITE, (x_pos,y_pos, self.cell_width, self.cell_height))
                    elif self.grid.getIndex(i, j) == -2:
                        pygame.draw.rect(self.screen, GRAY, (x_pos,y_pos, self.cell_width, self.cell_height))
                    else:
                        pygame.draw.rect(self.screen, GREEN, (x_pos,y_pos, self.cell_width, self.cell_height))                    
                        
            # set fps
            self.clock.tick(20)

            # display changes
            pygame.display.flip()

        #quit game
        pygame.quit()
        return self.grid.matrix

def inList(value, list):
    for item in list:
        if value == item:
            return True
    return False

def h(cell1,cell2):
    x1,y1=cell1
    x2,y2=cell2

    # distance = x1-y1 if x1 - y1 > y1-x1 else y1-x1

    return abs(x1-x2) + abs(y1-y2) 

class Solve:
    def __init__(self, grid, rows, cols, start, end):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end
        self.frontier = []
        self.path = [start]
   
    def adjacent_list(self):
        curr = self.path[-1]
        left = (curr[0]-1, curr[1])
        right = (curr[0]+1, curr[1])
        below = (curr[0], curr[1]+1)
        above = (curr[0], curr[1]-1)
        self.adjacent = [left, right, below, above]

        for i in range(len(self.adjacent)):
            try:
                if self.adjacent[i][0] < 0 or self.adjacent[i][1] < 0:
                    self.adjacent[i] = None
                elif self.adjacent[i][0] > self.rows or self.adjacent[i][1] > self.cols:
                    self.adjacent[i] = None
                elif self.grid[self.adjacent[i][0]][self.adjacent[i][1]] < 0:
                    self.adjacent[i] = None
                elif inList(self.adjacent[i], self.path):
                    self.adjacent[i] = None
            except:
                self.adjacent[i] = None
        self.adjacent = list(filter(None, self.adjacent))

    def select_from_frontier(self):
        self.frontier = self.frontier + self.adjacent
        next = self.frontier[0]
        idx = 0

        for i in range(len(self.frontier)):
            if h(self.frontier[i], self.end)<=h(next, self.end):
                next = self.frontier[i]
                idx = i 

        self.frontier.pop(idx)
        self.path.append(next)
        self.grid[next[0]][next[1]] = 1


if __name__ == "__main__":
    maze = Maze(255,255)
    
    matrix = maze.run()

    solve = Solve(matrix, matrix.shape[0],matrix.shape[1], (0, 0), (matrix.shape[0]-1,matrix.shape[1]-1))

    while solve.path[-1] != solve.end:
        solve.adjacent_list()
        solve.select_from_frontier()
        
        if len(solve.path) > 100:
            break

    maze2 = Maze(255, 255)
    maze2.grid.matrix = solve.grid
    maze2.run()