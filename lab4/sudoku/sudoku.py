"""Excercise 3: Simulated Annealing for Sudoku Solver"""
import math
import random
import time

from util import (find_empty_cells, generate_neighbor_state_random,
                  initialize_board, load_board)
from vis import (plot_board, plot_convergence, plot_vs_empty_cells,
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


def simulated_annealing(orig_board, empty, initial_temp=1000, cooling_rate=0.999,
                         min_temp=0.0001):
    """Performs simulated annealing process for Sudoku solving."""
    current_board = initialize_board(orig_board, empty)
    current_cost = calculate_cost(current_board)
    best_board = current_board.copy()
    best_cost = current_cost

    temp = initial_temp
    iteration = 0
    stagnation_count = 0

    iterations_history = []
    cost_history = []

    start_time = time.time()

    while temp > min_temp and current_cost > 0 and stagnation_count < 2000:
        neighbor_board = generate_neighbor_state_random(current_board, empty)
        neighbor_cost = calculate_cost(neighbor_board)
        previous_cost = current_cost

        delta_cost = neighbor_cost - current_cost

        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            current_board = neighbor_board
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_board = current_board.copy()
                best_cost = current_cost

        if current_cost != previous_cost:
            stagnation_count = 0
        else:
            stagnation_count += 1

        temp *= cooling_rate
        iteration += 1

        iterations_history.append(iteration)
        cost_history.append(current_cost)

    elapsed_time = time.time() - start_time

    return best_board, iteration, iterations_history, cost_history, elapsed_time


if __name__ == "__main__":
    test_files = []
    results = []

    for name in range(20):
        test_files.append(f"./boards/{name}.txt")

    for file_path in test_files:
        original_board = load_board(file_path)
        empty_cells = find_empty_cells(original_board)
        empty_count = len(empty_cells)

        solution, iterations, iter_history, cost_hist, el_time =\
            simulated_annealing(original_board, empty_cells)

        final_cost = calculate_cost(solution)
        valid_solution = final_cost == 0
        results.append({
            'final_cost': final_cost,
            'empty_count': empty_count,
            'iterations': iterations,
        })

        print(f"\nFile: {file_path}")
        print(f"Empty cell amount: {empty_count}")
        print("Original board:")
        print_board(original_board)

        plot_board(original_board, f"{empty_count} empty cells - original board")

        print(f"Iterations: {iterations}")
        print(f"Execution time: {el_time:.2f}s")

        if valid_solution:
            print("\nFound solution:")
            print_board(solution)

            plot_board(solution, f"{empty_count} empty cells - correct solution")

        else:
            print("\nNo correct solution found. Best solution:")
            print_board(solution)

            plot_board(solution, f"{empty_count} empty cells - best solution (cost: {final_cost})")

            print(f"Cost: {final_cost}")

        if len(iter_history) > 0:
            plot_convergence(iter_history, cost_hist, empty_count)
        else:
            print("Correct solution found while initializing the board.")

    final_costs = [r['final_cost'] for r in results]
    empty_counts = [r['empty_count'] for r in results]
    iteration_counts = [r['iterations'] for r in results]

    plot_vs_empty_cells(iteration_counts, empty_counts, 'Iterations')
    plot_vs_empty_cells(final_costs, empty_counts, 'Final cost')
