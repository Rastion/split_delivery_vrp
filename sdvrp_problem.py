import math
import random
from qubots.base_problem import BaseProblem
import os

class SDVRPProblem(BaseProblem):
    """
    Split Delivery Vehicle Routing Problem (SDVRP)

    In this problem a fleet of identical trucks (each with capacity Q) must serve
    a set of customers with given demands. Trucks start and end their routes at a
    common depot. Unlike the classic CVRP, deliveries may be split among several trucks.
    
    The instance file uses a DIMACS-like format:
      - First line: two integers: the number of customers and the truck capacity.
      - Second line: a list of customer demands.
      - Next: two numbers representing the depot coordinates.
      - Then for each customer, two numbers representing the customer’s coordinates.
    
    A candidate solution is represented as a dictionary with two keys:
      - "routes": a list of length nb_trucks (where nb_trucks is set equal to nb_customers),
                   each element is a list of customer indices (0-indexed) that are visited by that truck.
      - "deliveries": a list (of the same length) of lists of delivered quantities.
                   For each truck k, deliveries[k] is a list of length nb_customers,
                   where the value at index i is the quantity delivered to customer i
                   (if customer i is visited in truck k’s route, it should be > 0; otherwise 0).
    
    The objective is to minimize the total distance traveled by all trucks.
    In addition, the following feasibility requirements must be met:
      1. Every customer must be visited by at least one truck.
      2. For each customer, the total delivered quantity (across trucks) must be at least its demand.
      3. For each truck, the total delivered quantity on its route must not exceed the truck capacity.
      
    Infeasible solutions are penalized by adding a large multiple of the total violation.
    """
    
    def __init__(self, instance_file):
        self.instance_file = instance_file
        (self.nb_customers,
         self.truck_capacity,
         self.distance_matrix,
         self.distance_depots,
         self.demands) = self.read_input_sdvrp(instance_file)
        # Following the Hexaly model, we set the number of trucks equal to the number of customers.
        self.nb_trucks = self.nb_customers
        self.penalty_multiplier = 1e6

    def read_elem(self, filename):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename) as f:
            return f.read().split()

    def read_input_sdvrp(self, filename):
        tokens = self.read_elem(filename)
        it = iter(tokens)
        nb_customers = int(next(it))
        truck_capacity = int(next(it))
        # Read customer demands
        demands = [int(next(it)) for _ in range(nb_customers)]
        # Read depot coordinates
        depot_x = float(next(it))
        depot_y = float(next(it))
        # Read customer coordinates
        customers_x = [float(next(it)) for _ in range(nb_customers)]
        customers_y = [float(next(it)) for _ in range(nb_customers)]
        distance_matrix = self.compute_distance_matrix(customers_x, customers_y)
        distance_depots = self.compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
        return nb_customers, truck_capacity, distance_matrix, distance_depots, demands

    def compute_distance_matrix(self, xs, ys):
        n = len(xs)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.compute_dist(xs[i], xs[j], ys[i], ys[j])
        return matrix

    def compute_distance_depots(self, depot_x, depot_y, xs, ys):
        n = len(xs)
        depot_dists = [0 for _ in range(n)]
        for i in range(n):
            depot_dists[i] = self.compute_dist(depot_x, xs[i], depot_y, ys[i])
        return depot_dists

    def compute_dist(self, xi, xj, yi, yj):
        exact = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        return int(math.floor(exact + 0.5))

    def evaluate_solution(self, candidate) -> float:
        """
        Given a candidate solution, compute its total distance traveled plus
        penalties for any feasibility violations.
        
        Candidate structure (dictionary):
          {
            "routes": list of lists of int, one list per truck (indices of customers visited),
            "deliveries": list of lists of float, one list per truck (delivered quantities for each customer)
          }
        """
        routes = candidate.get("routes", [])
        deliveries = candidate.get("deliveries", [])
        penalty = 0
        total_distance = 0

        # For each truck, if the route is nonempty, compute route distance and check truck capacity.
        for route, delivery in zip(routes, deliveries):
            if route:
                # Compute route distance: depot->first customer + last customer->depot +
                # sum over consecutive customer-to-customer distances.
                d = self.distance_depots[route[0]] + self.distance_depots[route[-1]]
                for i in range(1, len(route)):
                    d += self.distance_matrix[route[i - 1]][route[i]]
                total_distance += d
                # Sum delivered quantity on this truck (only for customers visited in its route)
                delivered_sum = sum(delivery[i] for i in route)
                if delivered_sum > self.truck_capacity:
                    penalty += (delivered_sum - self.truck_capacity)
            else:
                # If a truck is not used, any nonzero delivery is a violation.
                if any(q > 0 for q in delivery):
                    penalty += sum(q for q in delivery)
        
        # Ensure that every customer is visited at least once.
        visited = [False] * self.nb_customers
        for route in routes:
            for cust in route:
                visited[cust] = True
        for i, v in enumerate(visited):
            if not v:
                penalty += self.demands[i]  # add a penalty proportional to the unmet visit

        # For each customer, ensure that total delivered quantity meets its demand.
        for i in range(self.nb_customers):
            total_delivered = sum(delivery[i] for delivery in deliveries)
            if total_delivered < self.demands[i]:
                penalty += (self.demands[i] - total_delivered)
        
        return total_distance + self.penalty_multiplier * penalty

    def random_solution(self):
        """
        Generates a simple feasible candidate solution.
        
        For this basic solution we:
          - Randomly permute the customers.
          - Assign each customer to a distinct truck (i.e. truck k gets one customer).
          - For each truck that is used, set the delivered quantity equal to the customer's demand.
          - For trucks not used (if any), use empty routes and zero deliveries.
        """
        perm = list(range(self.nb_customers))
        random.shuffle(perm)
        routes = []
        deliveries = []
        # We assume number of trucks equals nb_customers.
        for k in range(self.nb_trucks):
            if k < len(perm):
                route = [perm[k]]
                delivery = [0] * self.nb_customers
                # Deliver the full demand for the customer served by this truck.
                delivery[perm[k]] = self.demands[perm[k]]
            else:
                route = []
                delivery = [0] * self.nb_customers
            routes.append(route)
            deliveries.append(delivery)
        return {"routes": routes, "deliveries": deliveries}
