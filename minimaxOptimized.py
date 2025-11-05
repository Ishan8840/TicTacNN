import random
import numpy as np
from functools import lru_cache


# winning combinations for Tic-Tac-Toe (indices board)
WIN_COMBOS = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
]

def check_winner_static(board):
    for a,b,c in WIN_COMBOS:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return None

# minimax algorithm with caching to compute the best move for ai
@lru_cache(maxsize=None)
def minimax_cached(board_tuple, is_maximizing):
    winner = check_winner_static(board_tuple)
    if winner == 1: return 1
    if winner == -1: return -1
    if 0 not in board_tuple: return 0

    if is_maximizing:
        best_score = float("-inf")
        for i, spot in enumerate(board_tuple):
            if spot == 0:
                new_board = list(board_tuple)
                new_board[i] = 1
                best_score = max(best_score, minimax_cached(tuple(new_board), False))
        return best_score
    else:
        best_score = float("inf")
        for i, spot in enumerate(board_tuple):
            if spot == 0:
                new_board = list(board_tuple)
                new_board[i] = -1
                best_score = min(best_score, minimax_cached(tuple(new_board), True))
        return best_score

class TicTacToe:
    def __init__(self):
        self.board = [0]*9
        self.ai = 1
        self.human = -1

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == 0]

    def make_move(self, position, player):
        if self.board[position] == 0:
            self.board[position] = player
            return True
        return False

    def is_board_full(self):
        return 0 not in self.board

    def check_winner(self):
        return check_winner_static(self.board)

    def game_over(self):
        return self.check_winner() is not None or self.is_board_full()

    # Compute the best move for the AI using minimax.
    def get_best_move(self):
        best_score = float("-inf")
        best_move = None
        board_tuple = tuple(self.board)

        for move in self.available_moves():
            new_board = list(board_tuple)
            new_board[move] = self.ai
            score = minimax_cached(tuple(new_board), False)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move


# Generate training data for AI by simulating games.
def generate_training_data(num_games):
    data = []

    for _ in range(num_games):
        game = TicTacToe()
        ai_turn = random.choice([True, False])

        while not game.game_over():
            if ai_turn:
                move = game.get_best_move()
                if move is not None:
                    data.append((np.array(game.board, dtype=int), move))
                    game.make_move(move, game.ai)
            else:
                if random.random() < 0.8:
                    move = game.get_best_move()
                else:
                    move = random.choice(game.available_moves())
                if move is not None:
                    game.make_move(move, game.human)

            ai_turn = not ai_turn

    print(f"Generated {len(data)} board states.")
    return data



if __name__ == "__main__":
    dataset = generate_training_data(2000)
    X = np.array([state for state, _ in dataset])
    y = np.array([move for _, move in dataset])

    np.save("X.npy", X)
    np.save("y.npy", y)
    print(len(dataset))
    print(dataset[:5])
