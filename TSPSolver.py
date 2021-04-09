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

INF = np.inf


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
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

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < INF:
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

    # time complexity: O(n) * O(n) = O(n^)
    # space complexity: O(n) + O(n) + O(n) = O(3n) = O(n)
    def greedy(self, time_allowance=60.0):
        route_found = False
        route = []  # Space: O(n)
        list_of_possible_start_cities = self._scenario.getCities().copy()  # Space: O(n)
        cities = self._scenario.getCities()  # Space: O(n)
        start_city = list_of_possible_start_cities.pop()
        city = start_city
        route.append(city)
        start_time = time.time()
        while route_found is False and (time.time() - start_time) < time_allowance:  # O(n)
            lowest_cost = math.inf
            lowest_city = None
            for neighbor in cities:  # O(n)
                if neighbor is city:
                    continue
                if city.costTo(neighbor) < lowest_cost and (neighbor not in route):
                    lowest_cost = city.costTo(neighbor)
                    lowest_city = neighbor
            if lowest_city is None:  # check to see if can't continue
                if city.costTo(start_city) < lowest_cost:  # check to see if we're done
                    route_found = True
                    best_sol_so_far = TSPSolution(route)
                else:
                    route.clear()
                    start_city = list_of_possible_start_cities.pop()
                    city = start_city
            else:  # we did find a lowest_city
                route.append(lowest_city)
                city = lowest_city

        end_time = time.time()
        results = {'route': best_sol_so_far.route, 'cost': best_sol_so_far.cost if route_found else math.inf,
                   'time': end_time - start_time, 'count': len(route), 'soln': best_sol_so_far, 'max': None,
                   'total': None, 'pruned': None}
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    # time complexity: worse case: O(n!) - average: O(p) * (O(log n) + (O(n) * (O(log n) + O(n^2) + O(log n)))) =
    #                                               O(p) * (O(log n) + O(n * n^2)) = O(p) * O(n^3) = O(pn^3)
    # space complexity: worse case: O(n!) - average: (p) * O(n^2 + n) = O(p * n^2)
    def branch_and_bound(self, time_allowance=60.0):
        count = pruned_states = 0
        max_heap_size = total_states = 1
        solution_to_beat = TSPSolution(self.greedy()['route'])  # O(n^2) for time, O(n) for space

        heap = []
        cities = self._scenario.getCities()
        state = SearchState([cities[0]], cities, 0)
        state.init_matrix()
        state.reduce_matrix()
        heappush(heap, state)

        start_time = time.time()
        # time complexity: worse case: O(n!) - average: O(p) * (O(log n) + (O(n) * (# O(log n) + O(n^2) + O(log n))))
        # space complexity: worse case: O(n!) - average: (p) * O(n^2 + n) = O(p * n^2)
        while (time.time() - start_time) < time_allowance and len(heap) > 0:

            # record the biggest heap size we've seen
            max_heap_size = len(heap) if len(heap) > max_heap_size else max_heap_size

            # get next state to analyze
            current_state = heappop(heap)  # O(log n)

            # if state is less costly
            if current_state.best_cost < solution_to_beat.cost:

                # if path contains all cities (make sure it's a valid solution)
                if len(current_state.route) == len(cities):
                    last_cost = current_state.route[-1].costTo(current_state.route[0])
                    current_state.best_cost += last_cost

                    # if state cost is better than our current best solution
                    if current_state.best_cost < solution_to_beat.cost:
                        solution_to_beat = TSPSolution(deepcopy(current_state.route))
                        count += 1

                # if out path doesn't contain all cities (make it a loop)
                else:
                    for city in cities:  # O(n)

                        # add cities that are not in our path
                        if not current_state.city_in_route(city):
                            total_states += 1
                            new_path = current_state.route.copy().append(city)  # add current city
                            new_state = SearchState(new_path, cities, current_state.best_cost)
                            new_state.cost_matrix = np.copy(current_state.cost_matrix)  # O(log n)
                            city1 = new_state.route[new_state.len() - 2]
                            city2 = new_state.route[new_state.len() - 1]
                            new_state.set_cities_to_infinity(city1._index, city2._index)  # O(1)
                            new_state.reduce_matrix()  # O(n^2)

                            # if the new state could be better than the current solution
                            if new_state.best_cost < solution_to_beat.cost:
                                heappush(heap, new_state)  # O(log n)
                            # if the new state can't beat the current solution then we prune
                            else:
                                pruned_states += 1

            # if there is not improvement (after evaluating the state) then we prune the state
            else:
                pruned_states += 1

        end_time = time.time()

        results = {'cost': solution_to_beat.cost, 'time': end_time - start_time, 'count': count,
                   'soln': solution_to_beat, 'max': max_heap_size, 'total': total_states,
                   'pruned': pruned_states + len(heap)}
        return results

    def fancy(self, time_allowance=60.0):
        pass


class SearchState:
    def __init__(self, path, cities, best_cost):
        super().__init__()
        self.cost_matrix = np.zeros(shape=(len(self.cities), len(self.cities)))
        self.best_cost = best_cost
        self.route = path
        self.cities = cities

    # comparison function
    # time complexity: O(1)
    # space complexity: O(1)
    def __lt__(self, value):
        if len(self.route) is not len(value.route):
            return len(self.route) > len(value.route)
        else:
            return self.best_cost < value.best_cost

    # initialize cost matrix for cities
    # time complexity: O(n) * O(n) * O(1) = O(n^2)
    # space complexity: O(n^2)
    def init_matrix(self):
        self.cost_matrix = np.zeros(shape=(len(self.cities), len(self.cities)))  # O(n^2)
        row_index = 0
        for fromCity in self.cities:  # O(n)
            col_index = 0
            for toCity in self.cities:  # O(n)
                self.cost_matrix[row_index][col_index] = fromCity.costTo(toCity)  # O(1)
                col_index += 1
            row_index += 1

    # time complexity: O(1)
    # space complexity: O(1)
    def __str__(self):
        return str(self.cost_matrix)

    # time complexity: O(1)
    # space complexity: O(1)
    def len(self):
        return len(self.route)

    # time complexity: O(n)
    # space complexity: O(1)
    def city_in_route(self, city):
        for val in self.route:  # Worst case: O(n)
            if val._index == city._index:  # O(1)
                return True
        return False

    # time complexity: O(1)
    # space complexity: O(1)
    def set_matrix(self, matrix):
        self.cost_matrix = matrix

    # gets the cost from two cities, while setting the row and column to infinity
    # time complexity: O(n) + O(n) + O(1) = O(2n + 1) = O(2n) = O(n)
    # space complexity: O(1)
    def set_cities_to_infinity(self, from_city, to_city):
        self.best_cost += self.cost_matrix[from_city][to_city]
        if self.best_cost != INF:
            self.set_row_to_infinity(from_city)  # O(n)
            self.set_column_to_infinity(to_city)  # O(n)
            self.set_city_to_infinity(from_city, to_city)  # O(1)

    # reduce matrix by reducing columns and rows
    # time complexity: O(n^2) + O(n^2) = O(2n^2) = O(n^2)
    # space complexity: O(1)
    def reduce_matrix(self):
        # skip reducing if best cost is infinity (it's not going to get better)
        if self.best_cost != INF:
            self.reduce_matrix_rows()  # O(n^2)
            self.reduce_matrix_columns()  # O(n^2)

    # reduce columns of the cost matrix by finding the minimum and reducing if > 0 and < INF
    # time complexity: O(n) * O(n + n) = O(n) * O(2n) = O(2n^2) = O(n^2)
    # space complexity: O(1)
    def reduce_matrix_rows(self):
        for row in range(len(self.cost_matrix)):  # O(n)
            minimum, minimum_index = self.find_row_minimum(row)  # O(n)
            if minimum > 0 and minimum != INF:
                self.reduce_row(row, minimum)  # O(k)
                self.best_cost += minimum

    # reduce columns of the cost matrix by finding the minimum and reducing if > 0 and < INF
    # time complexity: O(n) * O(n + n) = O(n) * O(2n) = O(2n^2) = O(n^2)
    # space complexity: O(1)
    def reduce_matrix_columns(self):
        for column in range(len(self.cost_matrix[0])):  # O(n)
            minimum, minimum_index = self.find_column_minimum(column)  # O(n)
            if minimum > 0 and minimum != INF:
                self.reduce_column(column, minimum)  # O(n)
                self.best_cost += minimum

    # reduce column by subtracting the cost and minimum val
    # time complexity: O(n)
    # space complexity: O(1)
    def reduce_row(self, row, minimum):
        for column in range(len(self.cost_matrix[row])):  # O(n)
            new_cost = self.cost_matrix[row][column] - minimum
            if new_cost == np.nan:
                self.cost_matrix[row][column] = INF
            else:
                self.cost_matrix[row, column] = new_cost

    # reduce column by subtracting the cost and minimum val
    # time complexity: O(n)
    # space complexity: O(1)
    def reduce_column(self, column, minimum):
        for row in range(len(self.cost_matrix)):  # O(n)
            new_cost = self.cost_matrix[row][column] - minimum
            if new_cost == np.nan:
                self.cost_matrix[row][column] = INF
            else:
                self.cost_matrix[row, column] = new_cost

    # find lowest value in the cost matrix row
    # time complexity: O(n)
    # space complexity: O(1)
    def find_row_minimum(self, row):
        minimum = self.cost_matrix[row][0]
        minimum_index = 0
        for column in range(len(self.cost_matrix[row])):  # O(n)
            if self.cost_matrix[row][column] < minimum:
                minimum = self.cost_matrix[row][column]
                minimum_index = column

        return minimum, minimum_index

    # find lowest value in the cost matrix column
    # time complexity: O(n)
    # space complexity: O(1)
    def find_column_minimum(self, column):
        min_index = 0
        minimum = self.cost_matrix[min_index][column]
        for row in range(len(self.cost_matrix)):  # O(n)
            if self.cost_matrix[row][column] < minimum:
                minimum = self.cost_matrix[row][column]
                min_index = row
        return minimum, min_index

    # set a row in the cost matrix to infinity
    # time complexity: O(n)
    # space complexity: O(1)
    def set_row_to_infinity(self, row):
        self.cost_matrix[row][:] = INF

    # set a column in the cost matrix to infinity
    # time complexity: O(n)
    # space complexity: O(1)
    def set_column_to_infinity(self, column):
        self.cost_matrix[:][column] = INF

    # set a given city's cost to infinity
    # time complexity: O(1)
    # space complexity: O(1)
    def set_city_to_infinity(self, row, col):
        self.cost_matrix[col][row] = INF
