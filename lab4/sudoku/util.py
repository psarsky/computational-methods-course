"""Utility functions for Sudoku solver."""
import random

import numpy as np


def load_board(file_path):
    """Loads a Sudoku board from a text file."""
    board = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            row = []
            for char in line.strip():
                if char == 'x':
                    row.append(0)  # 0 represents an empty cell
                else:
                    row.append(int(char))
            board.append(row)
    return np.array(board)


def find_empty_cells(board):
    """Finds all empty cells in the Sudoku board."""
    empty_cells = []
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                empty_cells.append((i, j))
    return empty_cells


def initialize_board(original_board, empty_cells):
    """Initializes the Sudoku board by filling empty cells with random values."""
    board = original_board.copy()

    for i, j in empty_cells:
        used_values = set(board[i, :]).union(set(board[:, j]))
        block_i, block_j = 3 * (i // 3), 3 * (j // 3)
        used_values = used_values.union(set(board[block_i:block_i+3, block_j:block_j+3].flatten()))

        available_values = list({1, 2, 3, 4, 5, 6, 7, 8, 9} - used_values)

        if available_values:
            board[i, j] = random.choice(available_values)
        else:
            board[i, j] = random.randint(1, 9)

    return board


def generate_neighbor_state_random(board, empty_cells):
    """Generates a neighbor board by changing one random empty cell to a different value."""
    neighbor_state = board.copy()

    i, j = random.choice(empty_cells)

    current_value = neighbor_state[i, j]
    possible_values = list(range(1, 10))
    possible_values.remove(current_value)
    neighbor_state[i, j] = random.choice(possible_values)

    return neighbor_state


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
