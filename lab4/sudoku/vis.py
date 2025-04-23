"""Visualization utilities for Sudoku solver."""
import matplotlib.pyplot as plt
import numpy as np


def print_board(board):
    """Displays the Sudoku board in a readable format."""
    for i in range(9):
        if i % 3 == 0 and i > 0:
            print("-" * 21)
        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j > 0:
                row_str += "| "
            row_str += str(board[i, j]) + " "
        print(row_str)


def plot_board(board):
    """Plots the Sudoku board."""
    _, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(board, cmap='Blues', alpha=0.5)

    repeated = np.zeros_like(board, dtype=bool)

    for i in range(9):
        values, counts = np.unique(board[i, :][board[i, :] != 0], return_counts=True)
        for value in values[counts > 1]:
            repeated[i, board[i, :] == value] = True

    for j in range(9):
        values, counts = np.unique(board[:, j][board[:, j] != 0], return_counts=True)
        for value in values[counts > 1]:
            repeated[board[:, j] == value, j] = True

    for box_i in range(3):
        for box_j in range(3):
            box = board[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3]
            values, counts = np.unique(box[box != 0], return_counts=True)
            for value in values[counts > 1]:
                box_repeated = box == value
                repeated[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3][box_repeated] = True

    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                if repeated[i, j]:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='red', alpha=0.5))
                ax.text(j, i, str(board[i, j]), va='center', ha='center', fontsize=20)

    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color='black', lw=lw)
        ax.axvline(i - 0.5, color='black', lw=lw)

    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_convergence(iterations_history, cost_history, empty_count):
    """Plots the convergence of the algorithm."""
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_history, cost_history)
    plt.title(f'Algorithm convergence for {empty_count} empty cells')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def plot_iters_vs_empty_cells(iteration_counts, empty_counts):
    """Plots the number of iterations vs. the number of empty cells."""
    plt.figure(figsize=(10, 6))
    plt.scatter(empty_counts, iteration_counts)
    plt.title('Iterations vs. empty cells')
    plt.xlabel('Empty cell amount')
    plt.ylabel('Iterations')
    plt.grid(True)

    if len(empty_counts) > 1:
        z = np.polyfit(empty_counts, iteration_counts, 1)
        p = np.poly1d(z)
        plt.plot(empty_counts, p(empty_counts), "r--")

    plt.show()
