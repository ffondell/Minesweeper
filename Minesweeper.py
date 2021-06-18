import numpy as np
from numpy import zeros
import random
from matplotlib import pyplot as plt

class Node:#Can initizalize a node with attributes of a square in the minesweeper grid
    def __init__(self, row, col):
        self.row = row
        self.col = col


def generateGrid(d, n):#generates a minesweeper grid with dimension d and mines n
    grid = np.empty(shape=(d,d),dtype='object')

    for i in range(n):
        randx = random.randint(0, d-1)
        randy = random.randint(0, d-1)
        while(grid[randx][randy]=='m'):
            randx = random.randint(0, d-1)
            randy = random.randint(0, d-1)
        grid[randx][randy]='m'
    return grid

def printNodes(nodes):#prints all nodes by coordinates for debug purposes
    for i in range(len(nodes)):
        print("\n"+str(nodes[i].row)+", "+str(nodes[i].col))

def getEducatedGuesses(knowledgeBase):
    d = len(knowledgeBase)
    guesses = []
    guesses.append(Node(0,0))
    guesses.append(Node(0,d-1))
    guesses.append(Node(d-1,d-1))
    guesses.append(Node(d-1,0))
    i = 1

    while(i!=d-1):
        guesses.append(Node(0,i))
        guesses.append(Node(i,d-1))
        guesses.append(Node(d-1,d-1-i))
        guesses.append(Node(d-1-i,0))
        i+=1

    return guesses

def queryGridImproved(mineGrid, mines):
    knowledgeBase = np.empty(shape=(len(mineGrid),len(mineGrid[0])),dtype='object')
    knowledgeBase.fill('?')
    round = 0
    revealedMines = 0
    guess = None
    noGuesses = False
    fringe = getEducatedGuesses(knowledgeBase)
    visited = []
    while('?' in knowledgeBase):
        if(round==0):#if its the first round, the guess will be a random coordinate
            guess = fringe[0]
            round+=1
            fringe.remove(guess)
        else:
            if(len(fringe)==0):#only fills fringe if empty
                for x in range(len(knowledgeBase)):#if not first round, just basic loop to find safe squares and reveal them
                    for y in range(len(knowledgeBase[0])):
                        if(knowledgeBase[x][y]=='0'):
                            """
                            fringe.extend(getHiddenNeighbors(knowledgeBase, Node(x,y), visited))
                            visited.extend(getHiddenNeighbors(knowledgeBase, Node(x,y), visited))
                            """
                            new = getHiddenNeighbors(knowledgeBase, Node(x,y), visited)
                            if new:
                                fringe = new.extend(fringe)
                                visited.extend(getHiddenNeighbors(knowledgeBase, Node(x,y), visited))#keep track of coordinates that are already added to fringe

            if(len(fringe)!=0):#if the fringe has safe guesses in it, pop them out
                guess = fringe[0]
                fringe.remove(guess)
            else:#if the fringe is still empty after attempting to be filled, a new random guess must be choosen to continue game
                randx = random.randint(0, len(mineGrid)-1)
                randy = random.randint(0, len(mineGrid)-1)
                while(knowledgeBase[randx][randy]!='?'):
                    randx = random.randint(0, len(mineGrid)-1)
                    randy = random.randint(0, len(mineGrid)-1)
                guess = Node(randx, randy)
        #print("\nCurrent Guess: "+str(guess.row)+", "+str(guess.col))

        if(mineGrid[guess.row][guess.col]=='m'):
            knowledgeBase[guess.row][guess.col]='m'
        else:
            if(oneTwoPunch(knowledgeBase)!=None):#add mine revealed by 1 2 punch to fringe
                theMine = oneTwoPunch(knowledgeBase)
                knowledgeBase[theMine.row][theMine.col]='m'
                revealedMines+=1
                #print("Found hidden mine with method")
            if(getClue(mineGrid, Node(guess.row,guess.col))-getClue(knowledgeBase, guess)==len(getNumHiddenNeighbors(knowledgeBase, guess))):
                revealedMines = revealedMines + len(getNumHiddenNeighbors(knowledgeBase, guess))
                neighbors = getNumHiddenNeighbors(knowledgeBase, guess)
                for l in range(len(getNumHiddenNeighbors(knowledgeBase, guess))):
                    knowledgeBase[neighbors[l].row][neighbors[l].col]='m'
                    if(neighbors[l] in fringe):
                        fringe.remove(neighbors[l])
            numNeighbors = numberNeighbors(knowledgeBase, guess)
            if((numNeighbors-getClue(mineGrid, Node(guess.row,guess.col)))-(numNeighbors-len(getHiddenNeighbors(knowledgeBase, guess, visited)))==len(getHiddenNeighbors(knowledgeBase, guess, visited))):
                fringe.extend(getHiddenNeighbors(knowledgeBase, guess, visited))
                visited.extend(getHiddenNeighbors(knowledgeBase, guess, visited))
            if(mineGrid[guess.row][guess.col]!='m'):
                knowledgeBase[guess.row][guess.col]=str(getClue(mineGrid, Node(guess.row,guess.col)))



        #print("\n")
        #print(knowledgeBase)
        #print("fringe"+str(printNodes(fringe)))
    return revealedMines/mines

def queryGrid(mineGrid, mines):#agent that determines the best square to query and queries it
    knowledgeBase = np.empty(shape=(len(mineGrid),len(mineGrid[0])),dtype='object')
    knowledgeBase.fill('?')
    round = 0
    revealedMines = 0
    guess = None
    noGuesses = False
    fringe = []
    visited = []
    while('?' in knowledgeBase):
        if(round==0):#if its the first round, the guess will be a random coordinate
            randx = random.randint(0, len(mineGrid)-1)
            randy = random.randint(0, len(mineGrid)-1)
            while(knowledgeBase[randx][randy]!='?'):
                randx = random.randint(0, len(mineGrid)-1)
                randy = random.randint(0, len(mineGrid)-1)
            guess = Node(randx, randy)
        else:
            if(len(fringe)==0):#only fills fringe if empty
                for x in range(len(knowledgeBase)):#if not first round, just basic loop to find safe squares and reveal them
                    for y in range(len(knowledgeBase[0])):
                        if(knowledgeBase[x][y]=='0'):
                            fringe.extend(getHiddenNeighbors(knowledgeBase, Node(x,y), visited))
                            visited.extend(getHiddenNeighbors(knowledgeBase, Node(x,y), visited))#keep track of coordinates that are already added to fringe
            if(len(fringe)!=0):#if the fringe has safe guesses in it, pop them out
                guess = fringe[0]
                fringe.remove(guess)
            else:#if the fringe is still empty after attempting to be filled, a new random guess must be choosen to continue game
                randx = random.randint(0, len(mineGrid)-1)
                randy = random.randint(0, len(mineGrid)-1)
                while(knowledgeBase[randx][randy]!='?'):
                    randx = random.randint(0, len(mineGrid)-1)
                    randy = random.randint(0, len(mineGrid)-1)
                guess = Node(randx, randy)
        #print("\nCurrent Guess: "+str(guess.row)+", "+str(guess.col))
        if(mineGrid[guess.row][guess.col]=='m'):
            knowledgeBase[guess.row][guess.col]='m'
        else:
            if(getClue(mineGrid, Node(guess.row,guess.col))-getClue(knowledgeBase, guess)==len(getNumHiddenNeighbors(knowledgeBase, guess))):
                revealedMines = revealedMines + len(getNumHiddenNeighbors(knowledgeBase, guess))
                neighbors = getNumHiddenNeighbors(knowledgeBase, guess)
                for l in range(len(getNumHiddenNeighbors(knowledgeBase, guess))):
                    knowledgeBase[neighbors[l].row][neighbors[l].col]='m'
                    if(neighbors[l] in fringe):
                        fringe.remove(neighbors[l])
            numNeighbors = numberNeighbors(knowledgeBase, guess)
            if((numNeighbors-getClue(mineGrid, Node(guess.row,guess.col)))-(numNeighbors-len(getHiddenNeighbors(knowledgeBase, guess, visited)))==len(getHiddenNeighbors(knowledgeBase, guess, visited))):
                fringe.extend(getHiddenNeighbors(knowledgeBase, guess, visited))
                visited.extend(getHiddenNeighbors(knowledgeBase, guess, visited))
            knowledgeBase[guess.row][guess.col]=str(getClue(mineGrid, Node(guess.row,guess.col)))
        round+=1
        #print(knowledgeBase)
        #print("fringe"+str(printNodes(fringe)))
    return revealedMines/mines

def numberNeighbors(grid, space):#gets number of neighbors for a given tile
    neighbors = 0
    if((space.row-1>=0)and(space.col-1>=0)):#check top left square is out of bounds
        neighbors+=1
    if((space.row-1>=0)):#check top middle square is out of bounds
        neighbors+=1
    if((space.row-1>=0)and(space.col+1<len(grid))):#check top right square is out of bounds
        neighbors+=1
    if((space.col-1>=0)):#check middle left square is out of bounds
        neighbors+=1
    if((space.col+1<len(grid))):#check middle right square is out of bounds
        neighbors+=1
    if((space.row+1<len(grid))and(space.col-1>=0)):#check bottom left square is out of bounds
        neighbors+=1
    if((space.row+1<len(grid))):#check bottom middle square is out of bounds
        neighbors+=1
    if((space.row+1<len(grid))and(space.col+1<len(grid))):#check bottom left square is out of bounds
        neighbors+=1
    return neighbors


def getHiddenAdjacentNeighbors(space, knowledgeBase):#gets adjacent neighbors with numbers only for the 1 2 punch
    neighbors = [];
    if((space.row-1>=0)):#check top middle square is out of bounds
        if(knowledgeBase[space.row-1][space.col]!='?' and knowledgeBase[space.row-1][space.col]!='m'):#check if top middle square is hidden
            neighbors.append(Node(space.row-1,space.col))
    if((space.col-1>=0)):#check middle left square is out of bounds
        if(knowledgeBase[space.row][space.col-1]!='?' and knowledgeBase[space.row][space.col-1]!='m'):#check if middle left square is hidden
            neighbors.append(Node(space.row,space.col-1))
    if((space.col+1<len(knowledgeBase))):#check middle right square is out of bounds
        if(knowledgeBase[space.row][space.col+1]!='?' and knowledgeBase[space.row][space.col+1]!='m'):#check if middle right square is hidden
            neighbors.append(Node(space.row,space.col+1))
    if((space.row+1<len(knowledgeBase))):#check bottom middle square is out of bounds
        if(knowledgeBase[space.row+1][space.col]!='?' and knowledgeBase[space.row+1][space.col]!='m'):#check if bottom middle square is hidden
            neighbors.append(Node(space.row+1,space.col))
    return neighbors

def oneTwoPunch(knowledgeBase):#returns the location of a single space that satisfies the 1 2 rule
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            if(knowledgeBase[x][y]=='1' or knowledgeBase[x][y]=='2'):
                space = Node(x,y)
                #print("Space: "+str(space.row)+", "+str(space.col))
                neighbors = getHiddenAdjacentNeighbors(space, knowledgeBase)
                if(knowledgeBase[x][y]=='1'):
                    #print("here1")
                    for n in range(len(neighbors)):
                        if(knowledgeBase[neighbors[n].row][neighbors[n].col]=='2'):
                            #print("matching neighbor at: "+str(neighbors[n].row)+", "+str(neighbors[n].col))
                            if(len(getNumHiddenNeighbors(knowledgeBase, space))==3 and len(getNumHiddenNeighbors(knowledgeBase, Node(neighbors[n].row, neighbors[n].col)))==3):

                                if(sameRows(getNumHiddenNeighbors(knowledgeBase, space), getNumHiddenNeighbors(knowledgeBase, Node(neighbors[n].row, neighbors[n].col)))):

                                    if(space.row-1==getNumHiddenNeighbors(knowledgeBase, space)[0].row):
                                        return Node(space.row-1,space.col+2)
                                    if(space.row+1==getNumHiddenNeighbors(knowledgeBase, space)[0].row):
                                        return Node(space.row+1,space.col+2)
                                if(sameCols(getNumHiddenNeighbors(knowledgeBase, space), getNumHiddenNeighbors(knowledgeBase, Node(neighbors[n].row, neighbors[n].col)))):

                                    if(space.col-1==getNumHiddenNeighbors(knowledgeBase, space)[0].col):
                                        return Node(space.row+2,space.col-1)
                                    if(space.col+1==getNumHiddenNeighbors(knowledgeBase, space)[0].col):
                                        return Node(space.row+2,space.col+1)
                if(knowledgeBase[x][y]=='2'):
                    #print("here2")
                    for n in range(len(neighbors)):
                        if(knowledgeBase[neighbors[n].row][neighbors[n].col]=='1'):
                            #print("matching neighbor at: "+str(neighbors[n].row)+", "+str(neighbors[n].col))
                            if(len(getNumHiddenNeighbors(knowledgeBase, space))==3 and len(getNumHiddenNeighbors(knowledgeBase, Node(neighbors[n].row, neighbors[n].col)))==3):

                                if(sameRows(getNumHiddenNeighbors(knowledgeBase, space), getNumHiddenNeighbors(knowledgeBase, Node(neighbors[n].row, neighbors[n].col)))):

                                    if(space.row-1==getNumHiddenNeighbors(knowledgeBase, space)[0].row):
                                        return Node(space.row-1,space.col-1)
                                    if(space.row+1==getNumHiddenNeighbors(knowledgeBase, space)[0].row):
                                        return Node(space.row+1,space.col-1)
                                if(sameCols(getNumHiddenNeighbors(knowledgeBase, space), getNumHiddenNeighbors(knowledgeBase, Node(neighbors[n].row, neighbors[n].col)))):

                                    if(space.col-1==getNumHiddenNeighbors(knowledgeBase, space)[0].col):
                                        return Node(space.row-1,space.col-1)
                                    if(space.col+1==getNumHiddenNeighbors(knowledgeBase, space)[0].col):
                                        return Node(space.row-1,space.col+1)
    return None

def sameRows(neighbors1, neighbors2):#checks if two spaces share a row of hidden spaces
    sameRows = 0
    repRow = neighbors1[0].row
    for i in range(len(neighbors1)):
        if(neighbors1[i].row==repRow and neighbors2[i].row==repRow):
            sameRows+=1
    if(sameRows==3):
        return True
    else:
        return False

def sameCols(neighbors1, neighbors2):#checks if two spaces share a row of hidden spaces
    sameCols = 0
    repCol = neighbors1[0].col
    for i in range(len(neighbors1)):
        if(neighbors1[i].col==repCol and neighbors2[i].col==repCol):
            sameCols+=1
    if(sameCols==3):
        return True
    else:
        return False

def inVisited(nodes, node):#check to see if a node exists in an array
    for i in range(len(nodes)):
        if((node.row==nodes[i].row) and (node.col==nodes[i].col)):
            return True
    return False

def getNumHiddenNeighbors(grid, space):#same as below but returns an int of of hidden neighbors without regard for whether it is in visited or not
    neighbors = [];
    if((space.row-1>=0)and(space.col-1>=0)):#check top left square is out of bounds
        if(grid[space.row-1][space.col-1]=='?' and grid[space.row-1][space.col]!='m'):#check if top left square is hidden
            neighbors.append(Node(space.row-1,space.col-1))
    if((space.row-1>=0)):#check top middle square is out of bounds
        if(grid[space.row-1][space.col]=='?' and grid[space.row-1][space.col]!='m'):#check if top middle square is hidden
            neighbors.append(Node(space.row-1,space.col))
    if((space.row-1>=0)and(space.col+1<len(grid))):#check top right square is out of bounds
        if(grid[space.row-1][space.col+1]=='?' and grid[space.row-1][space.col+1]!='m'):#check if top right square hidden
            neighbors.append(Node(space.row-1,space.col+1))
    if((space.col-1>=0)):#check middle left square is out of bounds
        if(grid[space.row][space.col-1]=='?' and grid[space.row][space.col-1]!='m'):#check if middle left square is hidden
            neighbors.append(Node(space.row,space.col-1))
    if((space.col+1<len(grid))):#check middle right square is out of bounds
        if(grid[space.row][space.col+1]=='?' and grid[space.row][space.col+1]!='m'):#check if middle right square is hidden
            neighbors.append(Node(space.row,space.col+1))
    if((space.row+1<len(grid))and(space.col-1>=0)):#check bottom left square is out of bounds
        if(grid[space.row+1][space.col-1]=='?' and grid[space.row+1][space.col-1]!='m'):#check if bottom left square is hidden
            neighbors.append(Node(space.row+1,space.col-1))
    if((space.row+1<len(grid))):#check bottom middle square is out of bounds
        if(grid[space.row+1][space.col]=='?' and grid[space.row+1][space.col]!='m'):#check if bottom middle square is hidden
            neighbors.append(Node(space.row+1,space.col))
    if((space.row+1<len(grid))and(space.col+1<len(grid))):#check bottom left square is out of bounds
        if(grid[space.row+1][space.col+1]=='?' and grid[space.row+1][space.col+1]!='m'):#check if bottom left square is hidden
            neighbors.append(Node(space.row+1,space.col+1))
    return neighbors

def getHiddenNeighbors(grid, space, visited):#get all hidden neighbors for querying, input is knowledgeBase
    neighbors = []
    if((space.row-1>=0)and(space.col-1>=0)):#check top left square is out of bounds
        if(grid[space.row-1][space.col-1]=='?' and grid[space.row-1][space.col-1]!='m'):#check if top left square is hidden
            toAdd = Node(space.row-1, space.col-1)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.row-1>=0)):#check top middle square is out of bounds
        if(grid[space.row-1][space.col]=='?' and grid[space.row-1][space.col]!='m'):#check if top middle square is hidden
            toAdd = Node(space.row-1, space.col)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.row-1>=0)and(space.col+1<len(grid))):#check top right square is out of bounds
        if(grid[space.row-1][space.col+1]=='?' and grid[space.row-1][space.col+1]!='m'):#check if top right square hidden
            toAdd = Node(space.row-1, space.col+1)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.col-1>=0)):#check middle left square is out of bounds
        if(grid[space.row][space.col-1]=='?' and grid[space.row][space.col-1]!='m'):#check if middle left square is hidden
            toAdd = Node(space.row, space.col-1)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.col+1<len(grid))):#check middle right square is out of bounds
        if(grid[space.row][space.col+1]=='?' and grid[space.row][space.col+1]!='m'):#check if middle right square is hidden
            toAdd = Node(space.row, space.col+1)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.row+1<len(grid))and(space.col-1>=0)):#check bottom left square is out of bounds
        if(grid[space.row+1][space.col-1]=='?' and grid[space.row+1][space.col-1]!='m'):#check if bottom left square is hidden
            toAdd = Node(space.row+1, space.col-1)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.row+1<len(grid))):#check bottom middle square is out of bounds
        if(grid[space.row+1][space.col]=='?' and grid[space.row+1][space.col]!='m'):#check if bottom middle square is hidden
            toAdd = Node(space.row+1, space.col)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    if((space.row+1<len(grid))and(space.col+1<len(grid))):#check bottom left square is out of bounds
        if(grid[space.row+1][space.col+1]=='?' and grid[space.row+1][space.col+1]!='m'):#check if bottom left square is hidden
            toAdd = Node(space.row+1, space.col+1)
            if(not(inVisited(visited, toAdd))):
                neighbors.append(toAdd)
    return neighbors

def getClue(grid, space):#helper method to get the clue for a queried square, input is mineGrid
    hiddenMines = 0
    if((space.row-1>=0)and(space.col-1>=0)):#check top left square is out of bounds
        if(grid[space.row-1][space.col-1]=='m'):#check if top left square is a mine
            hiddenMines+=1
    if((space.row-1>=0)):#check top middle square is out of bounds
        if(grid[space.row-1][space.col]=='m'):#check if top middle square is a mine
            hiddenMines+=1
    if((space.row-1>=0)and(space.col+1<len(grid))):#check top right square is out of bounds
        if(grid[space.row-1][space.col+1]=='m'):#check if top right square is a mine
            hiddenMines+=1
    if((space.col-1>=0)):#check middle left square is out of bounds
        if(grid[space.row][space.col-1]=='m'):#check if middle left square is a mine
            hiddenMines+=1
    if((space.col+1<len(grid))):#check middle right square is out of bounds
        if(grid[space.row][space.col+1]=='m'):#check if middle right square is a mine
            hiddenMines+=1
    if((space.row+1<len(grid))and(space.col-1>=0)):#check bottom left square is out of bounds
        if(grid[space.row+1][space.col-1]=='m'):#check if bottom left square is a mine
            hiddenMines+=1
    if((space.row+1<len(grid))):#check bottom middle square is out of bounds
        if(grid[space.row+1][space.col]=='m'):#check if bottom middle square is a mine
            hiddenMines+=1
    if((space.row+1<len(grid))and(space.col+1<len(grid))):#check bottom left square is out of bounds
        if(grid[space.row+1][space.col+1]=='m'):#check if bottom left square is a mine
            hiddenMines+=1
    return hiddenMines

def plotSimple(numTrials, dim):#runs trials of minesweeper games with simple algorithm and plots x-mine density(in mines per board) by y-percent of safe revealed mines
    for mines in range(1,101):
        sum = 0.0
        for t in range(numTrials):
            sum = sum + queryGrid(generateGrid(dim, mines), mines)
        plt.plot(mines, sum/numTrials, "ob")
    plt.title("Mine Density vs Average Final Score (Simple Algorithm)")
    plt.xlabel("Mine Density")
    plt.ylabel("Average Final Score")
    plt.show()

def plotImproved(numTrials, dim):#runs trials of minesweeper games with simple algorithm and plots x-mine density(in mines per board) by y-percent of safe revealed mines
    finalSum = 0.0
    for mines in range(1,101):
        sum = 0.0
        for t in range(numTrials):
            sum = sum + queryGrid(generateGrid(dim, mines), mines)
        finalSum = finalSum + sum
        plt.plot(mines, sum/numTrials, "ob")
    plt.title("Mine Density vs Average Final Score (Improved Algorithm with Better Selection Mechanism)")
    plt.xlabel("Mine Density")
    plt.ylabel("Average Final Score")
    plt.show()

    return finalSum/(100.0*float(numTrials))


#knowledgeBase[1][0]='1'
#knowledgeBase[1][2]='1'

"""
#top 1 2
knowledgeBase[1][0]='0'
knowledgeBase[2][0]='0'
knowledgeBase[2][1]='0'
knowledgeBase[2][2]='0'
knowledgeBase[2][3]='0'
knowledgeBase[1][3]='0'

knowledgeBase[1][1]='2'
knowledgeBase[1][2]='1'
"""
"""
#right 1 2
knowledgeBase[0][2]='0'
knowledgeBase[0][3]='0'
knowledgeBase[1][3]='0'
knowledgeBase[2][2]='0'
knowledgeBase[3][2]='0'
knowledgeBase[1][2]='0'
knowledgeBase[3][3]='0'


knowledgeBase[1][3]='2'
knowledgeBase[2][3]='1'
"""
"""
#bottom 1 2
knowledgeBase[3][0]='0'
knowledgeBase[2][0]='0'
knowledgeBase[2][1]='0'
knowledgeBase[2][2]='0'
knowledgeBase[2][3]='0'
knowledgeBase[3][3]='0'

knowledgeBase[3][1]='2'
knowledgeBase[3][2]='1'
"""

"""
#left 1 2
knowledgeBase[0][1]='0'
knowledgeBase[0][2]='0'
knowledgeBase[1][2]='0'
knowledgeBase[2][2]='0'
knowledgeBase[3][2]='0'
knowledgeBase[3][1]='0'


knowledgeBase[1][1]='2'
knowledgeBase[2][1]='1'
"""

#print(knowledgeBase)
#mineSpace = oneTwoPunch(knowledgeBase)
#print("Mine at: "+str(mineSpace.row)+", "+str(mineSpace.col))
#printNodes(getHiddenAdjacentNeighbors(Node(1,2), knowledgeBase))
#grid = generateGrid(5,5)

#guesses = getEducatedGuesses(grid)
#print(grid)
#print(grid)
#plotSimple(10, 10)
print(plotImproved(10, 10))

#print(str(queryGridImproved(grid, 10)))
