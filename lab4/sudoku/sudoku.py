"""Excercise 3: Simulated Annealing for Sudoku Solver"""
import math
import random
import time

from util import (find_empty_cells, generate_neighbor_state_random,
                  initialize_board, load_board)
from vis import (plot_board, plot_convergence, plot_iters_vs_empty_cells,
                 print_board)


def calculate_cost(board):
    """Calculates the cost as the sum of digit repetitions in rows, columns, and 3x3 blocks."""
    cost = 0

    for i in range(9):
        row = board[i, :]
        cost += 9 - len(set(row))

    for j in range(9):
        col = board[:, j]
        cost += 9 - len(set(col))

    for block_i in range(0, 9, 3):
        for block_j in range(0, 9, 3):
            block = board[block_i:block_i+3, block_j:block_j+3].flatten()
            cost += 9 - len(set(block))

    return cost


def simulated_annealing(original_board, empty_cells, initial_temp=1000, cooling_rate=0.999,
                         min_temp=0.0001):
    """Performs simulated annealing process for Sudoku solving."""
    current_board = initialize_board(original_board, empty_cells)
    current_cost = calculate_cost(current_board)
    best_board = current_board.copy()
    best_cost = current_cost

    temp = initial_temp
    iteration = 0

    iterations_history = []
    cost_history = []

    start_time = time.time()

    while temp > min_temp and current_cost > 0:
        neighbor_board = generate_neighbor_state_random(current_board, empty_cells)
        neighbor_cost = calculate_cost(neighbor_board)

        delta_cost = neighbor_cost - current_cost

        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            current_board = neighbor_board
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_board = current_board.copy()
                best_cost = current_cost

        temp *= cooling_rate
        iteration += 1

        iterations_history.append(iteration)
        cost_history.append(current_cost)

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.2f}s")
    print(f"Empty cell amount: {len(empty_cells)}")
    print(f"Iterations: {iteration}")

    return best_board, iteration, iterations_history, cost_history


def test_multiple_boards(file_paths):
    """
    Tests the algorithm on multiple boards and analyzes the dependency of the number of
    iterations on the number of empty cells.
    """
    results = []

    for file_path in file_paths:
        original_board = load_board(file_path)
        empty_cells = find_empty_cells(original_board)
        empty_count = len(empty_cells)

        solution, iterations, iter_history, cost_history = simulated_annealing(original_board, empty_cells)

        valid_solution = calculate_cost(solution) == 0
        results.append({
            'file': file_path,
            'empty_count': empty_count,
            'iterations': iterations,
            'valid_solution': valid_solution
        })

        print(f"\nFile: {file_path}")
        print(f"Empty cell amount: {empty_count}")
        print(f"Iterations: {iterations}")
        print(f"Correct solution: {'Yes' if valid_solution else 'No'}")

        if valid_solution:
            print("\nFound solution:")
            print_board(solution)
        else:
            print("\nNo correct solution found. Best solution:")
            print_board(solution)
            print(f"Cost: {calculate_cost(solution)}")

        plot_convergence(iter_history, cost_history, empty_count)

    empty_counts = [r['empty_count'] for r in results]
    iteration_counts = [r['iterations'] for r in results]

    plot_iters_vs_empty_cells(iteration_counts, empty_counts)


def test_single_board(file_path):
    """Solves a single Sudoku board."""
    original_board = load_board(file_path)
    empty_cells = find_empty_cells(original_board)

    print(f"Empty cell amount: {len(empty_cells)}")
    print("Original board:")
    print_board(original_board)
    plot_board(original_board)

    solution, _, iter_history, cost_history = simulated_annealing(
        original_board, empty_cells
    )

    print("\nFound solution:")
    print_board(solution)
    plot_board(solution)
    valid = calculate_cost(solution) == 0
    print(f"Correct solution: {'Yes' if valid else 'No'}")

    if not valid:
        print(f"Final cost: {calculate_cost(solution)}")

    plot_convergence(iter_history, cost_history, len(empty_cells))

    return solution, valid


if __name__ == "__main__":
    test_single_board("./boards/0.txt")

    test_files = []

    for name in range(20):
        test_files.append(f"./boards/{name}.txt")

    test_multiple_boards(test_files)
