import tkinter as tk  # For creating the GUI
from tkinter import ttk  # For themed tk widgets
import matplotlib.pyplot as plt  # For plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding matplotlib in tkinter
import numpy as np  # For numerical operations
import random  # For random selections in the genetic algorithm
import time  # For timing the execution
import statistics  # For calculating standard deviation

# Function to plot the tour on the given axes
def plot_tour(cities, tour, ax):
    ax.clear()  # Clear previous plot
    x = [cities[i].x for i in tour]  # Extract x-coordinates
    y = [cities[i].y for i in tour]  # Extract y-coordinates
    ax.plot(x, y, 'o-', markersize=5)  # Plot cities and connections
    ax.plot([x[0], x[-1]], [y[0], y[-1]], 'ro-')  # Connect the last city to the first
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Tour')

# Class to represent a city
class City:
    def __init__(self, x, y):
        self.x = x  # x-coordinate
        self.y = y  # y-coordinate

# Function to calculate Euclidean distance between two cities
def euclidean_distance(city1, city2):
    return np.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)

# Function to load cities from a TSP file
def load_tsp_file(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                coords = line.strip().split()
                x, y = float(coords[0]), float(coords[1])
                cities.append(City(x, y))
            except ValueError:
                continue  # Skip lines that can't be parsed as coordinates
    return cities

# Class implementing the GA
class TSP_GA:
    def __init__(self, cities, pop_size=100, elite_size=20, mutation_rate=0.01):
        self.cities = cities
        self.pop_size = pop_size  # Size of the population
        self.elite_size = elite_size  # Number of elite individuals to keep
        self.mutation_rate = mutation_rate  # Probability of mutation

    # Create initial population
    def create_initial_population(self):
        return [np.random.permutation(len(self.cities)).tolist() for _ in range(self.pop_size)]

    # Calculate fitness of a route (inverse of total distance)
    def calculate_fitness(self, route):
        return 1 / self.route_distance(route)

    # Calculate total distance of a route
    def route_distance(self, route):
        total_distance = 0
        for i in range(len(route)):
            from_city = self.cities[route[i]]
            to_city = self.cities[route[(i + 1) % len(route)]]
            total_distance += euclidean_distance(from_city, to_city)
        return total_distance

    # Select parents using tournament selection
    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(enumerate(population)), 5)
            winner = max(tournament, key=lambda x: fitnesses[x[0]])
            selected.append(winner[1])
        return selected

    # Perform ordered crossover
    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]
        remaining = [item for item in parent2 if item not in child]
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining.pop(0)
        return child

    # Perform partially mapped crossover
    def partially_mapped_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))

        # Copy the mapping section
        child[start:end] = parent1[start:end]

        # Create the mapping
        mapping = dict(zip(parent1[start:end], parent2[start:end]))

        # Fill in the remaining elements
        for i in range(size):
            if i < start or i >= end:
                candidate = parent2[i]
                while candidate in child:
                    candidate = mapping[candidate]
                child[i] = candidate

        return child

    # Perform swap mutation
    def swap_mutation(self, route):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # Evolve the population by selecting parents, performing crossover and mutation
    def evolve_population(self, population, crossover_method):
        fitnesses = [self.calculate_fitness(route) for route in population]
        elite = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)[:self.elite_size]
        elite = [x[1] for x in elite]
        
        selected = self.tournament_selection(population, fitnesses)
        children = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                if crossover_method == "OX":
                    child1 = self.ordered_crossover(selected[i], selected[i + 1])
                    child2 = self.ordered_crossover(selected[i + 1], selected[i])
                else:  # PMX
                    child1 = self.partially_mapped_crossover(selected[i], selected[i + 1])
                    child2 = self.partially_mapped_crossover(selected[i + 1], selected[i])
                children.extend([child1, child2])
        mutated = [self.swap_mutation(child) for child in children]
        return elite + mutated[:self.pop_size - self.elite_size]

    # Run one step of the genetic algorithm
    def run_step(self, population, crossover_method):
        population = self.evolve_population(population, crossover_method)
        best_route = max(population, key=self.calculate_fitness)
        best_distance = self.route_distance(best_route)
        return population, best_route, best_distance

# Class for the GUI application
class TSPGuiApp:
    def __init__(self, root, filename):
        self.root = root
        self.root.title("TSP Solver - Genetic Algorithm")
        
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Set up matplotlib figure and canvas
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.cities = load_tsp_file(filename)
        self.ga = TSP_GA(self.cities)

        self.create_widgets()
        self.initialize_plots()

    # Create GUI widgets
    def create_widgets(self):
        ttk.Label(self.control_frame, text="Crossover Method:").pack(anchor="w")
        self.crossover_var = tk.StringVar(value="OX")
        ttk.Radiobutton(self.control_frame, text="Ordered Crossover", variable=self.crossover_var, value="OX").pack(anchor="w")
        ttk.Radiobutton(self.control_frame, text="Partially Mapped Crossover", variable=self.crossover_var, value="PMX").pack(anchor="w")

        ttk.Label(self.control_frame, text="Mutation Rate:").pack(anchor="w", pady=(10, 0))
        self.mutation_rate_var = tk.DoubleVar(value=0.01)
        ttk.Scale(self.control_frame, from_=0.001, to=0.1, variable=self.mutation_rate_var, orient=tk.HORIZONTAL).pack(fill="x")

        ttk.Label(self.control_frame, text="Number of Runs:").pack(anchor="w", pady=(10, 0))
        self.num_runs_var = tk.IntVar(value=1)
        ttk.Entry(self.control_frame, textvariable=self.num_runs_var, width=5).pack(anchor="w")

        self.run_button = ttk.Button(self.control_frame, text="Run Experiments", command=self.run_experiments)
        self.run_button.pack(pady=10)

        self.status_label = ttk.Label(self.control_frame, text="Ready to run")
        self.status_label.pack()
        
        self.stats_frame = ttk.Frame(self.control_frame)
        self.stats_frame.pack(pady=10)
        self.total_time_label = ttk.Label(self.stats_frame, text="Total Time: ")
        self.total_time_label.pack()
        self.min_distance_label = ttk.Label(self.stats_frame, text="Min Distance: ")
        self.min_distance_label.pack()
        self.max_distance_label = ttk.Label(self.stats_frame, text="Max Distance: ")
        self.max_distance_label.pack()
        self.avg_distance_label = ttk.Label(self.stats_frame, text="Avg Distance: ")
        self.avg_distance_label.pack()
        self.std_dev_label = ttk.Label(self.stats_frame, text="Std Deviation: ")
        self.std_dev_label.pack()

    # Initialize plots
    def initialize_plots(self):
        # Plot initial random tour
        initial_tour = list(range(len(self.cities))) + [0]
        plot_tour(self.cities, initial_tour, self.ax1)
        self.ax1.set_title("Initial Random Tour")

        # Plot empty improvement curve
        self.ax2.plot([], [])
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Best Tour Distance')
        self.ax2.set_title('GA Improvement Over Generations')
        self.canvas.draw()

    # Run experiments
    def run_experiments(self):
        self.run_button.config(state=tk.DISABLED)
        self.status_label.config(text="Running experiments...")
        self.root.update()

        crossover_method = self.crossover_var.get()
        mutation_rate = self.mutation_rate_var.get()
        num_runs = self.num_runs_var.get()

        print(f"\n--- Starting {num_runs} experiments ---")
        print(f"Crossover Method: {crossover_method}")
        print(f"Mutation Rate: {mutation_rate}")

        all_best_distances = []
        run_times = []

        total_start_time = time.time()

        for run in range(num_runs):
            self.status_label.config(text=f"Running experiment {run + 1}/{num_runs}")
            self.root.update()

            run_start_time = time.time()
            population = self.ga.create_initial_population()
            best_distances = []

            for generation in range(1000):
                population, best_route, best_distance = self.ga.run_step(population, crossover_method)
                best_distances.append(best_distance)

                if generation % 100 == 0:
                    self.update_plots(best_route, best_distances)
                    self.root.update()

            all_best_distances.append(best_distances)
            run_end_time = time.time()
            run_time = run_end_time - run_start_time
            run_times.append(run_time)

            # Print results for this run
            print(f"\nRun {run + 1} Results:")
            print(f"  Final Best Distance: {best_distances[-1]:.2f}")
            print(f"  Improvement: {best_distances[0] - best_distances[-1]:.2f}")
            print(f"  Improvement Percentage: {((best_distances[0] - best_distances[-1]) / best_distances[0]) * 100:.2f}%")
            print(f"  Run Time: {run_time:.2f} seconds")

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        self.plot_experiment_results(all_best_distances)
        
        # Calculate and display statistics
        all_final_distances = [distances[-1] for distances in all_best_distances]
        min_distance = min(all_final_distances)
        max_distance = max(all_final_distances)
        avg_distance = statistics.mean(all_final_distances)
        std_dev = statistics.stdev(all_final_distances) if len(all_final_distances) > 1 else 0

        print("\n--- Overall Results ---")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Best Distance: {min_distance:.2f}")
        print(f"Worst Distance: {max_distance:.2f}")
        print(f"Average Distance: {avg_distance:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Min Run Time: {min(run_times):.2f} seconds")
        print(f"Max Run Time: {max(run_times):.2f} seconds")
        print(f"Average Run Time: {statistics.mean(run_times):.2f} seconds")

        #GUI labels
        self.total_time_label.config(text=f"Total Time: {total_time:.2f} seconds")
        self.min_distance_label.config(text=f"Min Distance: {min_distance:.2f}")
        self.max_distance_label.config(text=f"Max Distance: {max_distance:.2f}")
        self.avg_distance_label.config(text=f"Avg Distance: {avg_distance:.2f}")
        self.std_dev_label.config(text=f"Std Deviation: {std_dev:.2f}")
        
        self.status_label.config(text="Experiments completed")
        self.run_button.config(state=tk.NORMAL)

    # Update plots during experiment
    def update_plots(self, best_route, best_distances):
        plot_tour(self.cities, best_route, self.ax1)
        self.ax1.set_title("Current Best Tour")
        
        self.ax2.clear()
        self.ax2.plot(range(len(best_distances)), best_distances)
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Best Tour Distance')
        self.ax2.set_title('GA Improvement Over Generations')
        self.ax2.set_title('GA Improvement Over Generations')
        self.canvas.draw()

    # Plot final experiment results
    def plot_experiment_results(self, all_best_distances):
        self.ax2.clear()
        for i, distances in enumerate(all_best_distances):
            self.ax2.plot(range(len(distances)), distances, label=f'Run {i+1}')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Tour Distance')
        self.ax2.set_title('GA Improvement Over Generations (All Runs)')
        self.ax2.legend()
        
        # x-axis stuff
        max_gen = len(all_best_distances[0])
        self.ax2.set_xticks(range(0, max_gen+1, max_gen//5))
        
        # y-axis stuff
        min_distance = min(min(distances) for distances in all_best_distances)
        max_distance = max(max(distances) for distances in all_best_distances)
        self.ax2.set_ylim(min_distance * 0.95, max_distance * 1.05)
        self.ax2.set_yticks(np.linspace(min_distance, max_distance, 6))
        
        # Adds grid
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Annotates improvement
        for i, distances in enumerate(all_best_distances):
            improvement = distances[0] - distances[-1]
            improvement_percent = (improvement / distances[0]) * 100
            self.ax2.annotate(f'Run {i+1} Improvement: {improvement:.2f} ({improvement_percent:.2f}%)', 
                              xy=(max_gen, distances[-1]), xytext=(10, 10),
                              textcoords='offset points', ha='left', va='bottom',
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                              arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
        self.canvas.draw()

# Main function
def main():
    root = tk.Tk()
    tsp_file = r'C:\Users\quarl\fall2024\CSE545\CSE545_Project4\Random100.tsp'
    app = TSPGuiApp(root, tsp_file)
    root.mainloop()

if __name__ == '__main__':
    main()