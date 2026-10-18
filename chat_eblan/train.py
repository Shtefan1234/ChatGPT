"""
Обучение модели Chat_eblan (минимальная версия M0).

Здесь ТОЛЬКО обучение: загрузка .bin-данных, батчи, цикл, чекпоинт.
Архитектура модели лежит в model.py, все настройки — в config.py.

Запуск:  python train.py   (из папки chat_eblan/)

Пайпeline перед запуском:
    python data.py     # подготовить data/train.bin, data/val.bin и токенизатор
    python train.py    # обучить и сохранить checkpoints/ckpt.pt
"""

import os
import numpy as np
import torch

import config
from model import GPTLanguageModel


# ---------------------------------------------------------------------------
# Данные: читаем .bin (uint16) через np.memmap — не грузим весь файл в RAM.
# np.memmap читает куски с диска на лету, экономит память на больших корпусах.
# ---------------------------------------------------------------------------
def load_tokens(path):
    """Читает .bin-файл токенов (uint16) и возвращает torch-тензор long."""
    data = np.memmap(path, dtype=np.uint16, mode='r')
    return torch.from_numpy(data.astype(np.int64))


train_data = load_tokens(config.TRAIN_BIN)
val_data = load_tokens(config.VAL_BIN)


def get_batch(split):
    """
    Генерирует один батч (x, y).

    x — вход: BATCH_SIZE случайных окон по BLOCK_SIZE токенов.
    y — цель: то же самое окно, сдвинутое на 1 токен вперёд
        (модель предсказывает следующий токен по предыдущим).

    split: 'train' или 'val'.
    """
    data = train_data if split == 'train' else val_data
    # случайные стартовые индексы окон (столько, сколько батчей)
    ix = torch.randint(len(data) - config.BLOCK_SIZE - 1, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i + config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + config.BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    return x, y


# ---------------------------------------------------------------------------
# Оценка loss: усредняем по EVAL_ITERS случайным батчам train и val.
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    """Считает средний loss на train и val (без обновления весов)."""
    out = {}
    model.eval()  # выключаем dropout во время оценки
    for split in ['train', 'val']:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # включаем dropout обратно
    return out


def save_checkpoint(model, optimizer, iter_num, path=config.CKPT_PATH):
    """
    Сохраняет чекпоинт: веса модели + её конфиг + номер итерации.
    Конфиг кладём внутрь, чтобы потом загрузить модель даже если config.py
    уже поменяли.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter_num,
        'config': {
            'vocab_size': config.VOCAB_SIZE_TOTAL,
            'block_size': config.BLOCK_SIZE,
            'n_embd': config.N_EMBD,
            'n_head': config.N_HEAD,
            'n_layer': config.N_LAYER,
            'dropout': config.DROPOUT,
        },
    }
    torch.save(ckpt, path)


def main():
    torch.manual_seed(config.SEED)

    # создаём модель из config и перемещаем на устройство
    model = GPTLanguageModel(
        vocab_size=config.VOCAB_SIZE_TOTAL,
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] устройство: {config.DEVICE}")
    print(f"[train] параметров: {n_params} ({n_params/1e6:.3f}M)")
    print(f"[train] train.bin={len(train_data)} токенов, val.bin={len(val_data)} токенов")
    print(f"[train] итераций всего: {config.MAX_ITERS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for iter_num in range(config.MAX_ITERS + 1):

        # каждые EVAL_INTERVAL итераций (и на последней) — оценка loss
        if iter_num % config.EVAL_INTERVAL == 0 or iter_num == config.MAX_ITERS:
            losses = estimate_loss(model)
            print(f"step {iter_num:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # остановились на последней итерации — только сохранили и вышли
        if iter_num == config.MAX_ITERS:
            break

        # берём батч
        xb, yb = get_batch('train')

        # forward: считаем предсказания и loss
        _, loss = model(xb, yb)
        # зануляем прошлые градиенты
        optimizer.zero_grad(set_to_none=True)
        # backward: считаем градиенты loss по всем весам
        loss.backward()
        # шаг оптимизатора: чуть подвинули веса в сторону уменьшения loss
        optimizer.step()

    # сохраняем итоговый чекпоинт
    save_checkpoint(model, optimizer, config.MAX_ITERS)
    print(f"[train] чекпоинт сохранён: {config.CKPT_PATH}")


if __name__ == "__main__":
    main()
