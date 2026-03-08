"""
player.py  —  Submit THIS file (along with requirements.txt).

TransformerPlayer uses GPT-2 Medium fine-tuned on 1M high-Elo chess positions.
Strategy: constrained decoding over all legal moves + aggressive play bonuses.
Fallback count is always 0 — only legal moves are returned.
"""

import torch
import chess
import random
from typing import Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from chess_tournament.players import Player

# Model config
# HF_MODEL_ID points to our fine-tuned GPT-2 Medium on HuggingFace.
# SEP_TOKEN must match exactly what was used during training in train.py.
HF_MODEL_ID = "mohammad-en/chess-engine-transformer"
SEP_TOKEN   = " MOVE "


class TransformerPlayer(Player):
    """
    Chess player based on GPT-2 Medium fine-tuned on high-Elo chess games.

    Architecture : GPT-2 Medium (decoder-only transformer, 345M parameters)
    Training data: 1,000,000 (FEN, move) pairs from adamkarvonen/chess_games
                   filtered for Elo >= 2000 and decisive results only (no draws)
    Training format: "<FEN> MOVE <uci_move><EOS>"
                     Loss computed ONLY on move tokens (FEN prefix masked)

    Move selection strategy:
      1. Build prefix : "<FEN> MOVE "
      2. Score every legal move via mean log-probability under the model
      3. Add tactical bonuses:
           +1.0  for captures     (aggressive play)
           +0.5  for checks       (tactical pressure)
           +10.0 for checkmate    (finish the game)
      4. Avoid draw by repetition (pick 2nd best move if needed)
      5. Return the highest-scoring legal move
      → Fallback count is always 0
    """

    def __init__(self, name: str = "TransformerPlayer",
                 model_id: str = HF_MODEL_ID):
        super().__init__(name)

        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{name}] Loading model '{model_id}' on {self.device}...")

        # Load tokenizer
        # Try loading tokenizer from HF repo first.
        # Falls back to base gpt2-medium tokenizer if repo has no tokenizer files
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        except Exception:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # match train.py

        # Load fine-tuned model weights from HuggingFace
        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()  # inference mode — disable dropout

        # Cache SEP token ids for use in scoring
        # (same SEP_TOKEN = " MOVE " used in training)
        self.sep_token = SEP_TOKEN
        self.sep_ids   = self.tokenizer.encode(
            SEP_TOKEN, add_special_tokens=False
        )

        total = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[{name}] Ready! ({total:.0f}M parameters on {self.device})")

    # Public API

    def get_move(self, fen: str) -> Optional[str]:
        """
        Return the best legal UCI move for a given FEN board state.

        Approach:
          - Score all legal moves using the fine-tuned GPT-2 model
          - Add capture/check/checkmate bonuses for aggressive play
          - Avoid draw by repetition by picking 2nd best if needed
          - Fallback to random only if model failed to load (should not happen)

        Args:
            fen: FEN string representing the current board state
        Returns:
            UCI move string (e.g. 'e2e4') — always a legal move
        """
        board       = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        # No legal moves : game is over
        if not legal_moves:
            return None

        # Safety fallback if model failed to load
        if self.model is None:
            return random.choice(legal_moves)

        # Build shared prefix used for all move candidates
        # Format mirrors training: "<FEN> MOVE <uci_move>"
        prefix = fen + self.sep_token
        scores = {}

        # Score every legal move and store in dict
        for uci in legal_moves:
            scores[uci] = self._score_move(prefix, uci, board)

        # Sort moves by score descending — best move first
        sorted_moves = sorted(scores.items(), key=lambda x: -x[1])
        best_move    = sorted_moves[0][0]

        # Repetition avoidance
        # If best move leads to draw by repetition, pick 2nd best instead.
        # This prevents the model from getting stuck in move loops so draw.
        try:
            board.push(chess.Move.from_uci(best_move))
            if board.is_repetition(2) and len(sorted_moves) > 1:
                board.pop()
                best_move = sorted_moves[1][0]  # use 2nd best move
            else:
                board.pop()
        except Exception:
            pass

        return best_move

    # Internal helpers 

    def _score_move(self, prefix: str, uci_move: str,
                    board: chess.Board) -> float:
        """
        Score a move using log-probability + tactical bonuses.

        Base score = mean log-probability of move tokens given FEN prefix.
        Bonuses are added to encourage aggressive, decisive play:
          +1.0  capture  → reward taking opponent pieces
          +0.5  check    → reward putting opponent king in check
          +10.0 checkmate→ always prefer winning moves

        Args:
            prefix  : "<FEN> MOVE " string (shared across all moves)
            uci_move: UCI move string to score (e.g. 'e2e4')
            board   : current chess.Board (used for tactical bonus checks)
        Returns:
            float score — higher is better
        """
        # Get base log-probability score from the language model
        base  = self._compute_logprob(prefix, uci_move)
        bonus = 0.0

        try:
            move = chess.Move.from_uci(uci_move)

            # Capture bonus — reward aggressive piece taking
            if board.is_capture(move):
                bonus += 1.0

            # Push move temporarily to check resulting board state
            board.push(move)

            # Check bonus — reward putting opponent king under attack
            if board.is_check():
                bonus += 0.5

            # Checkmate bonus — always prefer game-winning moves
            if board.is_checkmate():
                bonus += 10.0

            # Restore board to original state
            board.pop()

        except Exception:
            pass

        return base + bonus

    def _compute_logprob(self, prefix: str, uci_move: str) -> float:
        """
        Compute mean log-probability of uci_move tokens given prefix.

        This mirrors exactly how sequences were formatted in train.py:
            "<FEN> MOVE <uci_move><EOS>"

        The score is normalized by move length so short moves (e.g. 'e2e4')
        and long moves (e.g. 'e7e8q' promotions) are fairly compared.

        Args:
            prefix  : "<FEN> MOVE " tokenized as context
            uci_move: move string whose probability we compute
        Returns:
            mean log-probability (float) — higher means model prefers this move
        """
        try:
            # Tokenize full sequence: prefix + move
            full_ids   = self.tokenizer.encode(
                prefix + uci_move, return_tensors="pt"
            ).to(self.device)

            # Tokenize prefix alone to find where move tokens start
            prefix_ids = self.tokenizer.encode(
                prefix, return_tensors="pt"
            ).to(self.device)

            prefix_len = prefix_ids.shape[1]          # number of prefix tokens
            move_len   = full_ids.shape[1] - prefix_len  # number of move tokens

            if move_len <= 0:
                return float("-inf")

            # Forward pass — get logits for all positions
            with torch.no_grad():
                logits    = self.model(full_ids).logits  # (1, seq_len, vocab_size)
                # Convert logits to log-probabilities
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

            # Sum log-probs of move tokens only
            # logits[i] predicts token at position i+1
            # so to score move token at position prefix_len+i
            # we use logits at position prefix_len-1+i
            score = 0.0
            for i in range(move_len):
                pos      = prefix_len - 1 + i
                token_id = full_ids[0, prefix_len + i]
                score   += log_probs[pos, token_id].item()

            # Normalize by move length (mean log-prob per token)
            return score / move_len

        except Exception:
            return float("-inf")


# Sanity test
# Run this file directly to verify the player works correctly.
# Expected output: a legal UCI move from the starting position.
if __name__ == "__main__":
    player    = TransformerPlayer("Mohammad Ekhtiyariynaghash")

    # Test on starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move      = player.get_move(start_fen)

    # Verify move is legal
    board = chess.Board(start_fen)
    legal = [m.uci() for m in board.legal_moves]

    print(f"\\nMove     : {move}")
    print(f"Is legal : {move in legal}")  # should always be True
