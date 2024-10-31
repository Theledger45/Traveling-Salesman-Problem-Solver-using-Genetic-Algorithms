# Traveling-Salesman-Problem-Solver-using-Genetic-Algorithms
A Python implementation of a Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP) with an interactive GUI visualization. This project demonstrates the application of evolutionary optimization techniques to find near-optimal solutions for route optimization challenges.
Features

Interactive GUI built with Tkinter and Matplotlib for real-time visualization
Two crossover methods: Ordered Crossover (OX) and Partially Mapped Crossover (PMX)
Dynamic visualization of tour improvements and generation progress
Comprehensive statistics and performance metrics
Support for standard TSP file format input
Real-time plotting of best tour and improvement curves

Performance

Achieves up to 56% improvement in route optimization
Ordered Crossover (OX) method achieved best distance of 1891.92
Partially Mapped Crossover (PMX) method achieved best distance of 2374.53
Average run time of ~25 seconds per experiment

Requirements

Python 3.8+
tkinter
matplotlib
numpy
statistics
random
time

Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/tsp-genetic-algorithm.git
cd tsp-genetic-algorithm

Install required packages:

bashCopypip install -r requirements.txt
Usage

Run the main script:

bashCopypython main.py

Using the GUI:

Select crossover method (OX or PMX)
Adjust mutation rate using the slider
Set number of runs
Click "Run Experiments" to start optimization
View results in real-time through the visualization



Input Format
The program accepts TSP files with the following format:
Copyx1 y1
x2 y2
x3 y3
...
where each line represents the coordinates of a city.
