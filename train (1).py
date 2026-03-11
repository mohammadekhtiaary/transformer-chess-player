# ── Sympy fix — MUST BE FIRST ─────────────────────────────────────
import types, sympy
if not hasattr(sympy, 'core'):
    sympy.core = types.ModuleType('sympy.core')
    sympy.core.symbol = types.ModuleType('sympy.core.symbol')
    sympy.core.symbol.Symbol = sympy.Symbol

# ── Imports ───────────────────────────────────────────────────────
import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)

# ── Find dataset path automatically ───────────────────────────────
def find_csv():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(dirname, filename)
                print(f"  Found: {full_path}")
                return full_path
    return None

DATA_FILE = find_csv()
if DATA_FILE is None:
    raise FileNotFoundError("No CSV found in /kaggle/input")
print(f"Using dataset: {DATA_FILE}")

# ── Config
MODEL_NAME   = "gpt2-medium"
OUTPUT_DIR   = "/kaggle/working/chess_engine_transformer"
HF_REPO      = "mohammad-en/chess-engine-transformer"  # ← new name

MAX_LENGTH   = 96
BATCH_SIZE   = 24         
GRAD_ACCUM   = 11
FP16         = True
EPOCHS       = 1
LR           = 1e-4
WARMUP_STEPS = 500
MAX_SAMPLES  = 1_000_000
SEP_TOKEN    = " MOVE "


class ChessDataset(Dataset):
    """
    Each sample: "<FEN> MOVE <uci_move><EOS>"
    Loss computed ONLY on move tokens — FEN prefix is masked.
    """

    def __init__(self, csv_file, tokenizer, max_samples=None):
        self.tokenizer = tokenizer
        self.samples   = []

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                text = row["fen"] + SEP_TOKEN + row["move"] + tokenizer.eos_token
                self.samples.append(text)

        print(f"  Loaded {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.samples[idx],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        labels     = input_ids.clone()
        sep_ids    = self.tokenizer.encode(SEP_TOKEN, add_special_tokens=False)
        sep_len    = len(sep_ids)
        seq        = input_ids.tolist()
        mask_until = 0

        for j in range(len(seq) - sep_len + 1):
            if seq[j: j + sep_len] == sep_ids:
                mask_until = j + sep_len
                break

        labels[:mask_until]         = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def print_gpu_stats(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU [{label}]: {used:.1f}GB / {total:.1f}GB used")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"TRAINING CONFIG — MAX 15GB")
    print(f"{'='*60}")
    print(f"  Device          : {device}")
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Effective batch : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  FP16            : {FP16}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  LR              : {LR}")
    print(f"  Warmup steps    : {WARMUP_STEPS}")
    print(f"  Max samples     : {MAX_SAMPLES:,}")
    print(f"  HF repo         : {HF_REPO}")

    if torch.cuda.is_available():
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────
    print("\\nLoading dataset...")
    dataset  = ChessDataset(DATA_FILE, tokenizer, max_samples=MAX_SAMPLES)
    split    = int(0.95 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds   = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    print(f"  Train : {len(train_ds):,}")
    print(f"  Val   : {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────
    print(f"\\nLoading {MODEL_NAME}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {total_params:.0f}M")
    print_gpu_stats("after model load")

    # ── Optimizer & Scheduler ─────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=FP16)

    print(f"\\n  Total optimizer steps : {total_steps:,}")
    print(f"  Warmup steps          : {WARMUP_STEPS}")

    # ── Training Loop ─────────────────────────────────────────────
    best_val_loss = float("inf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\\n{'='*60}")
        print(f"EPOCH {epoch+1} / {EPOCHS}")
        print(f"{'='*60}")

        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(enabled=FP16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM

            if step % 200 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch+1} | Step {step:>5}/{len(train_loader)} "
                      f"| Loss: {avg:.4f}")

            # ── OOM safety check every 500 steps ──────────────
            if step % 500 == 0 and torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                if used / total > 0.92:    # over 92% → reduce batch
                    print(f"  High GPU usage {used:.1f}/{total:.1f}GB — watch out!")

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                with autocast(enabled=FP16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        print(f"\\nEpoch {epoch+1} complete | Val Loss: {val_loss:.4f}")
        print_gpu_stats(f"epoch {epoch+1} end")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Best model saved to {OUTPUT_DIR}/")
        else:
            print(f"  ✗ No improvement (best: {best_val_loss:.4f})")

    print(f"\\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Saved to      : {OUTPUT_DIR}/")
    print(f"{'='*60}")

    # ── Push to HuggingFace ───────────────────────────────────────
    try:
        from huggingface_hub import login
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        login(token=token)
        model     = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)
        tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
        model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)
        print(f"  ✓ Live at: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"  (HuggingFace push skipped: {e})")
        print(f"  Model saved locally at: {OUTPUT_DIR}/")


# ── Run ───────────────────────────────────────────────────────────
train()