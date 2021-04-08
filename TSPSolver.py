#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
from heapq import heappop, heappush
from copy import deepcopy
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		routeFound = False
		route = []
		listOfPossibleStartCities = self._scenario.getCities().copy()
		cities = self._scenario.getCities()
		startCity = listOfPossibleStartCities.pop()
		city = startCity
		route.append(city)
		start_time = time.time()
		while routeFound is False:
			lowestCost = math.inf
			lowestCity = None
			for neighbor in cities:
				if neighbor is city:
					continue
				if city.costTo(neighbor) < lowestCost and (neighbor not in route):
					lowestCost = city.costTo(neighbor)
					lowestCity = neighbor
			if lowestCity is None:  # check to see if can't continue
				if city.costTo(startCity) < lowestCost:  # check to see if we're done
					routeFound = True
					bssf = TSPSolution(route)
				else:
					route.clear()
					startCity = listOfPossibleStartCities.pop()
					city = startCity
			# route.append(city)
			else:  # We did find a lowestCity
				route.append(lowestCity)
				city = lowestCity

		end_time = time.time()
		results['route'] = bssf.route
		results['cost'] = bssf.cost if routeFound else math.inf
		results['time'] = end_time - start_time
		results['count'] = len(route)
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		startIndex = 0
		startCity = cities[startIndex]
		currentCity = startCity
		results = self.greedy()
		route = results['route']
		bssf = TSPSolution(route)
		foundTour = True
		heap = []
		iState = TSPState([cities[0]], cities, 0)
		iState.initMatrix()
		iState.reduceMatrix()
		heappush(heap, iState)
		start_time = time.time()
		maxHeapSize = 1
		totalStates = 1
		prunedStates = 0
		print("starting b&b")
		while (time.time() - start_time) < time_allowance and len(
				heap) > 0:  # searches state while there are potentially better paths, worst case O(n!), but approximates to O(n^k), where k = total states - pruned states
			if len(heap) > maxHeapSize:  # if current heapsize is greater than max size so far, set max size
				maxHeapSize = len(heap)  # so far to current heap size

			currentState = heappop(heap)  # pop best potential solution off queue
			if currentState.bestCost < bssf.cost:  # if the state cost is potentially better than current cost, continue
				if len(currentState.path) == len(
						cities):  # if every city is in the path, verify that the path is a cycle
					lastCost = currentState.path[-1].costTo(currentState.path[0])
					currentState.bestCost += lastCost
					if currentState.bestCost < bssf.cost:  # if state path is a cycle, set best cost and best path to state path and cost
						bssf = TSPSolution(deepcopy(currentState.path))
						count += 1
				else:
					for city in cities:
						if not currentState.inPath(
								city):  # visit every city that has not been visited by the current path
							totalStates += 1
							newPath = currentState.path.copy()
							newPath.append(city)
							newState = TSPState(newPath, cities, currentState.bestCost)
							newState.costMatrix = np.copy(currentState.costMatrix)
							city1 = newState.path[newState.len() - 2]
							city2 = newState.path[newState.len() - 1]
							newState.coverCities(city1._index, city2._index)
							newState.reduceMatrix()
							if newState.bestCost < bssf.cost:  # if state is potentially better than current best, push onto heap, else, prune state
								heappush(heap, newState)
							else:
								prunedStates += 1
			else:  # if state cost is worse than best cost, prune state
				prunedStates += 1

		end_time = time.time()
		print("ending b&b")
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxHeapSize
		results['total'] = totalStates
		results['pruned'] = prunedStates + len(heap)
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass


class TSPState:
	def __init__(self, path, cities, bestCost):
		super().__init__()
		self.bestCost = bestCost
		self.path = path
		self.cities = cities

	"""
        comparison function used by the heapq library. Initially compares the path length, choosing the longer path. 
        If the paths are the same length, then the current cost is used to compare heap values
        O(1) 
    """

	def __lt__(self, value):
		if len(self.path) is not len(value.path):
			return len(self.path) > len(value.path)
		else:
			return self.bestCost < value.bestCost

	"""         
        Initializes cost matrix to the costs between cities.
        O(n^2)
    """

	def initMatrix(self):
		self.costMatrix = np.zeros(shape=(len(self.cities), len(self.cities)))
		row_index = 0
		col_index = 0
		for fromCity in self.cities:
			col_index = 0
			for toCity in self.cities:
				self.costMatrix[row_index][col_index] = fromCity.costTo(toCity)
				col_index += 1
			row_index += 1

	def __str__(self):
		return str(self.costMatrix)

	def len(self):
		return len(self.path)

	def inPath(self, city):
		for val in self.path:
			if val._index == city._index:
				return True
		return False

	def setMatrix(self, matrix):
		self.costMatrix = matrix

	"""
        Grabs the travel cost from one city to the other. The from row, the to column, and the value at (toCity, fromCity) 
        are set to infinity. If cost is infinity, this operation is skipped because we are throwing out this state anyways
        O(2n + 1)
    """

	def coverCities(self, fromCity, toCity):
		self.bestCost += self.costMatrix[fromCity][toCity]
		if self.bestCost != np.inf:
			self.infRow(fromCity)
			self.infCol(toCity)
			self.infPair(fromCity, toCity)

	"""
        Performs row and column reduction operations on the cost matrix. If best cost is infinity, this operation is skipped because 
        the best cost will still be infinity, meaning the state will never enter the queue anyways
        O(kn^2) # where k is the number of cities
    """

	def reduceMatrix(self):
		if self.bestCost != np.inf:
			self.reduceMatrixRows()
			self.reduceMatrixCols()

	"""
        Performs a row reduction on every row by subtracting the minimum row value from every value in the row. This function also increments the bestCost 
        value. Operation is skipped if min value is 0 or infinity, due to the redundancy of such an operation. 
        O(kn + k) #  where k is the number of cities
    """

	def reduceMatrixRows(self):
		for row in range(len(self.costMatrix)):
			min, minIndex = self.findRowMin(row)
			if min != np.inf and min > 0:
				self.reduceRow(row, min, minIndex)
				self.bestCost += min

	"""
        Performs a col reduction by subtracting the minimum col value from every value in the col. This function also increments the bestCost 
        value. Operation is skipped if min value is 0 or infinity, due to the redundancy of such an operation. 
        O(kn + k) #  where k is the number of cities
    """

	def reduceMatrixCols(self):
		for col in range(len(self.costMatrix[0])):
			min, minIndex = self.findColMin(col)
			if min != np.inf and min > 0:
				self.reduceCol(col, min, minIndex)
				self.bestCost += min

	"""
        This function actually performs the row reduction, subtracting the min value from the row
        O(n)
    """

	def reduceRow(self, row, min, minIndex):
		for col in range(len(self.costMatrix[row])):
			newCost = self.costMatrix[row][col] - min
			if newCost == np.nan:
				self.costMatrix[row][col] = np.inf
			else:
				self.costMatrix[row, col] = newCost

	"""
        This function actually performs the column reduction, subtracting the min vale from the column
        O(n)
    """

	def reduceCol(self, col, min, minIndex):
		for row in range(len(self.costMatrix)):
			newCost = self.costMatrix[row][col] - min
			if newCost == np.nan:
				self.costMatrix[row][col] = np.inf
			else:
				self.costMatrix[row, col] = newCost

	"""
        Iterates through a row to find a minimum value and it's location
        O(n)
    """

	def findRowMin(self, row):
		min = self.costMatrix[row][0]
		minIndex = 0
		for col in range(len(self.costMatrix[row])):
			if self.costMatrix[row][col] < min:
				min = self.costMatrix[row][col]
				minIndex = col

		return min, minIndex

	"""
        Iterates through a column to find a minimum value and it's location
        O(n)
    """

	def findColMin(self, col):
		min = self.costMatrix[0][col]
		minIndex = 0
		row = 0
		for row in range(len(self.costMatrix)):
			if self.costMatrix[row][col] < min:
				min = self.costMatrix[row][col]
				minIndex = row
		return min, minIndex

	"""
        Iterates through a row and sets each value to infinity
        O(n)
    """

	def infRow(self, row):
		for i in range(len(self.costMatrix[row])):
			self.costMatrix[row][i] = np.inf

	"""
        Iterates through a column and sets each value to infinity
        O(n)
    """

	def infCol(self, col):
		for i in range(len(self.costMatrix)):
			self.costMatrix[i][col] = np.inf

	"""
        Sets the cost from the column city to the row city to infinity
        O(1)
    """

	def infPair(self, row, col):
		self.costMatrix[col][row] = np.inf