from copy import deepcopy
import pygame
from pygame.locals import *
import numpy as np
import random as rd
from dataclasses import dataclass


@dataclass
class Maze_Point:
    width: int
    height: int
    g_value: int

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

def inList2(value, list):
    for item in list:
        if value == (item.width, item.height):
            return True
    return False

class Grid:
    def __init__(self,row_length, col_length):
        self.matrix = self.create_maze(row_length)

    def create_maze(self, val):
        n=1
        p=0.5
        size=val

        # 1 (head) N, 0 (tail) E
        # np.random.seed(42)
        grid = np.random.binomial(n,p, size=(size,size))
        # print(grid)

        processed_grid = self.preprocess_grid(grid, size)
        # print(processed_grid)

        output = self.carve_maze(processed_grid, size)
        # print(output)

        return output

    def preprocess_grid(self, grid:np.ndarray, size:int) -> np.ndarray:
        # fix first row and last column to avoid digging outside the maze external borders
        first_row = grid[0]
        first_row[first_row == 1] = 0
        grid[0] = first_row
        for i in range(1,size):
            grid[i,size-1] = 1
        return grid

    def carve_maze(self, grid:np.ndarray, size:int) -> np.ndarray:
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

class Game:
   def __init__(self, Init_values_dict):

    # screen dimensions
    self.screen_w = Init_values_dict["Screen_Width"]
    self.screen_h = Init_values_dict["Screen_Height"]

    # cell dimensions
    self.cell_width = Init_values_dict["Cell_Width"]
    self.cell_height = Init_values_dict["Cell_Height"]
    self.cell_margin = Init_values_dict["Cell_Margin"]

    self.color_for_1 = Init_values_dict["Color for 1"]
    self.color_for_2 = Init_values_dict["Color for 2"]

    # init game and set screen dims
    pygame.init()

    # matrix for cells
    self.grid = Grid(
        self.screen_w // (self.cell_width+self.cell_margin), 
        self.screen_w // (self.cell_height+self.cell_margin) 
        )
    self.grid.setBound(-2)


    self.screen_w = self.screen_w * 3 - 10
    self.screen_h = self.screen_h * 3 - 10
    self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))

    # setup variables
    self.screen.fill(WHITE)
    self.background = BLACK
    self.running = True
    self.clock = pygame.time.Clock()
    self.drag=False
    self.clear = False

    A_star_init_values = {
        "Grid": self.grid.matrix,
        "Rows": self.grid.matrix.shape[0],
        "Columns": self.grid.matrix.shape[1],
        "Start": (0,0),
        "End": (self.grid.matrix.shape[0]-1, self.grid.matrix.shape[1]-1),
        "Color": 1
    }

    self.solve_for_end = A_star_matrix_solver(A_star_init_values)

    A_star_init_values["Start"] = (self.grid.matrix.shape[0]-1,self.grid.matrix.shape[1]-1)
    A_star_init_values["End"] = (0,0)
    A_star_init_values["Color"] = 2


    self.solve_for_start = A_star_matrix_solver(A_star_init_values)

   def solve_grid(self, algrotithm):
        if (algrotithm.path[-1].width, algrotithm.path[-1].height) != algrotithm.end and not algrotithm.stop:
                    if len(algrotithm.path) < 1000:
                        algrotithm.adjacent_list()
                        algrotithm.select_from_frontier()
                        self.grid.matrix = algrotithm.grid

   def inList2(value, list):
       for item in list:
        if value == (item.width, item.height):
            return True
        return False

   def are_algorithms_touching(self, algorithm):
        curr_solve_end = algorithm.path[-1]
        try:
            if self.grid.getIndex(curr_solve_end.width+1, curr_solve_end.height) > 0:
                if not inList2((curr_solve_end.width+1, curr_solve_end.height), algorithm.path):
                    return True
        except:
            pass

        try:
            if self.grid.getIndex(curr_solve_end.width-1, curr_solve_end.height) > 0:
                if not inList2((curr_solve_end.width-1, curr_solve_end.height), algorithm.path):
                    return True
        except:
            pass
                
        try:
            if self.grid.getIndex(curr_solve_end.width, curr_solve_end.height+1) > 0:
                if not inList2((curr_solve_end.width, curr_solve_end.height+1), algorithm.path):
                    return True
        except:
            pass

        try:
            if self.grid.getIndex(curr_solve_end.width, curr_solve_end.height-1) > 0:
                if not inList2((curr_solve_end.width, curr_solve_end.height-1), algorithm.path):
                    return True
        except:
            pass

        return False

   def Solve_Maze(self, grid, algorithm):
        if algorithm == "bidirectional":
            self.solve_for_start.start = (self.solve_for_start.cols-1, self.solve_for_start.rows-1)
            self.solve_for_end.start = (0,0)
        
        self.solve_for_end.set_path()
        self.solve_for_start.set_path()
        self.solve_for_end.grid = grid
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

            if algorithm == "double":
                self.solve_grid(self.solve_for_end)
                self.solve_for_start.grid = self.grid.matrix
                self.solve_grid(self.solve_for_start)
            elif algorithm == "single":
                self.solve_grid(self.solve_for_end)
            elif algorithm == "bidirectional":
                self.solve_for_end.end = (self.solve_for_start.path[-1].width,self.solve_for_start.path[-1].height)
                self.solve_for_end.start = (self.solve_for_start.path[-1].width, self.solve_for_start.path[-1].height)
                self.solve_grid(self.solve_for_end)

                if self.are_algorithms_touching(self.solve_for_end):
                    self.solve_for_end.stop = True
                    self.solve_for_start.stop = True

                self.solve_for_start.grid = self.grid.matrix
                self.solve_grid(self.solve_for_start)

                if self.are_algorithms_touching(self.solve_for_start):
                    self.solve_for_end.stop = True
                    self.solve_for_start.stop = True
                

            # create grid
            for Column in range(self.screen_h // (self.cell_height+self.cell_margin)):
                for Row in range(self.screen_w // (self.cell_width+self.cell_margin)):
                    y_pos = Column * (self.cell_height+self.cell_margin) +self.cell_margin
                    x_pos = Row * (self.cell_width+self.cell_margin) + self.cell_margin


            # Black if cell is 1 white otherwise
                    if self.grid.getIndex(Column, Row) == -1:
                        pygame.draw.rect(self.screen, BLACK, (x_pos,y_pos, self.cell_width, self.cell_height))
                    elif self.grid.getIndex(Column, Row) == 0:
                        pygame.draw.rect(self.screen, WHITE, (x_pos,y_pos, self.cell_width, self.cell_height))
                    elif self.grid.getIndex(Column, Row) == -2:
                        pygame.draw.rect(self.screen, GRAY, (x_pos,y_pos, self.cell_width, self.cell_height))
                    elif self.grid.getIndex(Column, Row) == 1:
                        pygame.draw.rect(self.screen, self.color_for_1, (x_pos,y_pos, self.cell_width, self.cell_height))                    
                    elif self.grid.getIndex(Column, Row) == 2:
                        pygame.draw.rect(self.screen, self.color_for_2, (x_pos,y_pos, self.cell_width, self.cell_height)) 
            # set fps
            self.clock.tick(20)

            # display changes
            pygame.display.flip()

        #quit game
        pygame.quit()
        print("It took", len(self.solve_for_end.path), "Iterations for algorithm solving for the end")
        print("It took", len(self.solve_for_start.path), "Iterations for algorithm solving for the end")

   def Init_Maze(self):
        start = (0, 0)
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
                        self.grid.setIndex(y_pos, x_pos, -1)
                        start = (y_pos, x_pos)

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
                        pygame.draw.rect(self.screen, BLACK, (x_pos,y_pos, self.cell_width, self.cell_height))                    

            # set fps
            self.clock.tick(20)

            # display changes
            pygame.display.flip()

        #quit game
        pygame.draw.rect(self.screen, WHITE, (start[0],start[1], self.cell_width, self.cell_height))
        pygame.quit()
        return self.grid.matrix, start

def inList(value, list):
    for item in list:
        if value == item:
            return True
    return False

class A_star_matrix_solver:
    def __init__(self, Solver_values_dict):
        self.grid = Solver_values_dict["Grid"]
        self.rows = Solver_values_dict["Rows"]
        self.cols = Solver_values_dict["Columns"]
        self.start = Solver_values_dict["Start"]
        self.end = Solver_values_dict["End"]
        self.color = Solver_values_dict["Color"]
        self.frontier = []
        
        self.stop = False
   
    def set_path(self):
        Start_Point = Maze_Point(self.start[0], self.start[1], 0)
        self.path = [Start_Point]

    def adjacent_list(self):
        Current_Point = self.path[-1]
        left = Maze_Point(Current_Point.width-1, Current_Point.height, Current_Point.g_value+1)
        right = Maze_Point(Current_Point.width+1, Current_Point.height, Current_Point.g_value+1)
        below = Maze_Point(Current_Point.width, Current_Point.height+1, Current_Point.g_value+1)
        above = Maze_Point(Current_Point.width, Current_Point.height-1, Current_Point.g_value+1)
        self.adjacent = [left, right, below, above]

        for i in range(len(self.adjacent)):
            try:
                if self.adjacent[i].width < 0 or self.adjacent[i].height < 0:
                    self.adjacent[i] = None
                elif self.adjacent[i].width > self.rows or self.adjacent[i].height > self.cols:
                    self.adjacent[i] = None
                elif self.grid[self.adjacent[i].width][self.adjacent[i].height] < 0:
                    self.adjacent[i] = None
                elif inList(self.adjacent[i], self.path):
                    self.adjacent[i] = None
                elif self.grid[self.adjacent[i].width][self.adjacent[i].height] != 0:
                    self.adjacent[i] = None
            except:
                self.adjacent[i] = None

        self.adjacent = list(filter(None, self.adjacent))

    def h(self, Current_Point, End_Point):
        return abs(Current_Point.width-End_Point[0]) + abs(Current_Point.height-End_Point[1]) 

    def f(self, Current_Point, End_Point):
        return Current_Point.g_value + self.h(Current_Point, End_Point)
    
    def select_from_frontier(self):
        self.frontier = self.frontier + self.adjacent
        try:
            Next_Point = self.frontier[0]
        except IndexError:
            self.stop = True
            return

        idx = 0
        if len(self.frontier) == 0:
            self.stop == True
        for i in range(len(self.frontier)):
            if inList(self.frontier[i], self.path):
                self.frontier[i] = None
                continue
            if self.f(self.frontier[i], self.end) < self.f(Next_Point, self.end):
                Next_Point = self.frontier[i]
                idx = i 

        self.frontier.pop(idx)
        self.frontier = list(filter(None, self.frontier))
        self.path.append(Next_Point)
        # print(Next_Point)
        self.grid[Next_Point.width][Next_Point.height] = self.color


if __name__ == "__main__":
    game_init_values = {
        "Screen_Width": 255,
        "Screen_Height": 255,
        "Cell_Width": 20,
        "Cell_Height": 20,
        "Cell_Margin": 5,
        "Color for 1": GREEN,
        "Color for 2": BLUE
    }    
    
    maze = Game(game_init_values)
    
    matrix, start = maze.Init_Maze()

    matrix2 = deepcopy(matrix)

    maze2 = Game(game_init_values)
    maze2.solve_for_end.start = start
    maze2.solve_for_end.set_path()
    maze2.solve_for_start.start = start
    maze2.solve_for_start.set_path()
    maze2.Solve_Maze(matrix, "double")

    maze3 = Game(game_init_values)
    maze3.solve_for_end.set_path()
    maze3.solve_for_start.set_path()
    maze3.Solve_Maze(matrix2, "bidirectional")