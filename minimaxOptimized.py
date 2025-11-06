import random
import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = [" " for _ in range(9)]
        self.human = "O"
        self.ai = "X"

    def print_board(self):
        for i in range(0, 9, 3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6:
                print("---------")

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == " "]
    
    def make_move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            return True
        return False
    
    def is_board_full(self):
        return " " not in self.board
    
    def check_winner(self):
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != " ":
                return self.board[i]
        
        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != " ":
                return self.board[i]
        
        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != " ":
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != " ":
            return self.board[2]
        
        return None
    
    def game_over(self):
        return self.check_winner() is not None or self.is_board_full()
    
    def minimax_for_player(self, depth, is_maximizing, player, opponent):
        winner = self.check_winner()
        if winner == player:
            return 1
        if winner == opponent:
            return -1
        if self.is_board_full():
            return 0
        
        if depth >= 6:
            return 0
        
        if is_maximizing:
            best_score = float("-inf")
            for move in self.available_moves():
                self.board[move] = player
                score = self.minimax_for_player(depth + 1, False, player, opponent)
                self.board[move] = " "
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float("inf")
            for move in self.available_moves():
                self.board[move] = opponent
                score = self.minimax_for_player(depth + 1, True, player, opponent)
                self.board[move] = " "
                best_score = min(score, best_score)
            return best_score
    
    def get_best_move_for_player(self, player):
        """Get best move for any player"""
        best_score = float("-inf")
        best_move = None
        opponent = self.human if player == self.ai else self.ai

        for move in self.available_moves():
            self.board[move] = player
            score = self.minimax_for_player(0, False, player, opponent)
            self.board[move] = " "

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
    
    def get_best_move(self):
        """Get best move for AI"""
        return self.get_best_move_for_player(self.ai)


def generate_training_data(n_games=5000):
    """Generate training data from tic-tac-toe games"""
    dataset = []
    
    for game_num in range(n_games):
        if game_num % 10 == 0:
            print(f"Generated {game_num} games...")
        
        game = TicTacToe()
        
        # Randomly decide who starts
        first_player = random.choice([game.ai, game.human])
        current_player = first_player
        
        while not game.game_over():
            # Encode from AI's perspective
            board_state = [1 if x == game.ai else -1 if x == game.human else 0 
                          for x in game.board]
            
            # Only collect data when AI is moving
            if current_player == game.ai:
                move = game.get_best_move()
                dataset.append((board_state.copy(), move))
                game.make_move(move, game.ai)
            else:

                move = random.choice(game.available_moves())

                game.make_move(move, game.human)
            
            # Switch players
            current_player = game.human if current_player == game.ai else game.ai
    
    print(f"Generated {n_games} games total!")
    return dataset


if __name__ == "__main__":
    print("Generating training data...")
    dataset = generate_training_data(5000)
    
    X = np.array([state for state, _ in dataset])
    y = np.array([move for _, move in dataset])
    
    print(f"\nDataset shape:")
    print(f"X (board states): {X.shape}")
    print(f"y (moves): {y.shape}")
    
    np.save("X.npy", X)
    np.save("y.npy", y)
    
    print("\nSaved to X.npy and y.npy")
    print("\nExample:")
    print(f"Board state: {X[0]}")
    print(f"Best move: {y[0]}")