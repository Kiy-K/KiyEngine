# training/dataset.py
import torch
from torch.utils.data import Dataset
import chess
import chess.pgn
from typing import List, Tuple
import random

def move_to_token(move: chess.Move, board: chess.Board) -> int:
    """ Maps a chess.Move to a token index, consistent with Rust core. """
    piece = board.piece_at(move.from_square)
    if piece is None:
        # This can happen in rare cases with faulty PGNs
        return 0

    piece_idx = move.promotion - 1 if move.promotion else piece.piece_type - 1
    if piece.color == chess.BLACK:
        piece_idx += 6

    to_square_idx = move.to_square
    return piece_idx * 64 + to_square_idx

def board_to_sequence(board: chess.Board, move_history: List[chess.Move]) -> List[int]:
    """ Converts a board state and move history into a sequence of tokens. """
    temp_board = chess.Board()
    sequence = []
    for move in move_history:
        sequence.append(move_to_token(move, temp_board))
        temp_board.push(move)
    return sequence

class ChessDataset(Dataset):
    """
    Dataset for loading chess games from a PGN file.
    Each item is a tuple of (input_sequence, policy_target, value_target).
    """
    def __init__(self, pgn_file_path: str, context_length: int = 16, augment: bool = True):
        self.pgn_file_path = pgn_file_path
        self.context_length = context_length
        self.augment = augment
        self.games = self._load_games()

    def _load_games(self) -> List[Tuple[List[chess.Move], float]]:
        """
        Loads games from the PGN file and extracts move history and game result.
        """
        processed_games = []
        pgn = open(self.pgn_file_path)

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                value = 1.0
            elif result_str == "0-1":
                value = -1.0
            elif result_str == "1/2-1/2":
                value = 0.0
            else:
                continue # Skip games with unknown results

            moves = list(game.mainline_moves())
            if len(moves) > self.context_length:
                processed_games.append((moves, value))

        return processed_games

    def __len__(self) -> int:
        # This is a simplification. A real implementation would count total positions.
        return len(self.games)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        moves, value = self.games[idx]

        # Select a random position from the game
        start_ply = random.randint(0, len(moves) - self.context_length - 1)
        end_ply = start_ply + self.context_length

        move_history = moves[start_ply:end_ply]
        target_move = moves[end_ply]

        # Create the board state at the start of the sequence
        board = chess.Board()
        for i in range(start_ply):
            board.push(moves[i])

        # Data Augmentation: Color Flipping
        if self.augment and random.random() < 0.5:
            board = board.mirror()
            move_history = [chess.Move.from_uci(m.uci()) for m in move_history] # Re-parse moves for new perspective
            target_move = chess.Move.from_uci(target_move.uci())
            value = -value

        # Convert to tensors
        input_sequence = board_to_sequence(board, move_history)
        policy_target = move_to_token(target_move, board)

        return (
            torch.tensor(input_sequence, dtype=torch.long),
            torch.tensor(policy_target, dtype=torch.long),
            torch.tensor([value], dtype=torch.float)
        )
