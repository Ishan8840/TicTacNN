import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

model = NeuralNetwork()
model.load_state_dict(torch.load("tictac_model.pth"))
model.eval()


def print_board(board):
    symbols = [' ', 'X', 'O']
    for i in range(0, 9, 3):
        print("|".join(symbols[board[j]] for j in range(i, i+3)))
        if i < 6:
            print("-----")
    print()

def check_winner(board):
    win_combos = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for combo in win_combos:
        a,b,c = combo
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0 
    return None

def ai_move(board):
    board_tensor = torch.FloatTensor(board).unsqueeze(0)
    with torch.no_grad():
        output = model(board_tensor)
        move = output.argmax(dim=1).item()
    if board[move] != 0:
        move = board.index(0)
    return move

board = [0]*9  # 0=empty, 1=player, 2=AI
print("Board positions: 0-8 (left to right, top to bottom)")

while True:
    # --- AI move first ---
    move = ai_move(board)
    print(f"AI moves to: {move}")
    board[move] = 2
    
    winner = check_winner(board)
    if winner is not None:
        print_board(board)
        if winner == 1:
            print("You win!")
        elif winner == 2:
            print("AI wins!")
        else:
            print("It's a draw!")
        break
    
    print_board(board)

    # --- Player move ---
    move = int(input("Your move (0-8): "))
    if board[move] != 0:
        print("Invalid move! Try again.")
        continue
    board[move] = 1

    winner = check_winner(board)
    if winner is not None:
        print_board(board)
        if winner == 1:
            print("You win!")
        elif winner == 2:
            print("AI wins!")
        else:
            print("It's a draw!")
        break
