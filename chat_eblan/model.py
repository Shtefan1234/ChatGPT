"""
Архитектура GPT-модели для Chat_eblan.

Здесь ТОЛЬКО архитектура (классы нейросети). Никаких данных, оптимизатора
и цикла обучения — всё это живёт в train.py.

Ключевое изменение против исходника: все размеры (vocab_size, block_size,
n_embd, n_head, n_layer, dropout) приходят в конструктор как АРГУМЕНТЫ,
а не берутся из глобальных переменных файла. Так модель переиспользуема
и её конфиг можно сохранить в чекпоинт.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# Это одна голова Self-Attention.
# Она позволяет каждому символу учитывать предыдущие символы контекста
# при формировании собственного представления.
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, block_size, n_embd, dropout):
        # head_size  — размерность ключей/запросов/значений ОДНОЙ головы
        # block_size — длина контекста (нужна для каузальной маски tril)
        # n_embd     — входная размерность (сколько признаков у каждого токена)
        # dropout    — вероятность выключения "взгляда" во время обучения
        super().__init__()
        # nn.Linear — обучаемое линейное преобразование.
        # У него есть матрица весов, которая изменяется во время обучения.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # создали треугольную матрицу из единиц (каузальная маска):
        # разрешает смотреть только на ПРЕДЫДУЩИЕ позиции, будущие запрещены.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute attention scores ("affinities")
        # k.transpose(-2,-1) = (B,T,hs) -> (B,hs,T)
        # q @ k.transpose(-2, -1) == (B,T,hs) @ (B,hs,T) -> (B,T,T)
        # Для каждой последовательности модель получает матрицу размером T × T.
        # Элемент этой матрицы показывает, насколько сильно одна позиция
        # сопоставляется с другой до применения маски и нормализации.
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # k.shape[-1]**-0.5 — масштабирование
        # Масштабирование помогает удерживать значения оценок внимания
        # в подходящем диапазоне перед softmax, что способствует
        # более стабильному обучению.

        # (B, T, T): где маска == 0 (будущее) присваиваем -бесконечность
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # (B, T, T): применяем softmax — позиции будущего получают
        # нулевую вероятность внимания
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        # (B, T, T) @ (B, T, hs) -> (B, T, hs): перемножаем матрицу вниманий
        # на матрицу значений
        out = wei @ v
        return out  # итог (B,T,hs)


# В нём создаются несколько независимых голов (по числу n_head):
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, block_size, n_embd, dropout):
        super().__init__()
        # запускаем num_heads голов внимания параллельно
        self.heads = nn.ModuleList([
            Head(head_size, block_size, n_embd, dropout) for _ in range(num_heads)
        ])
        # линейный слой для возврата к исходной размерности n_embd
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # рандомно отключаем часть нейронов
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # каждая голова выдаёт head_size признаков, склеиваем их по каналу
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,num_heads*head_size)
        out = self.dropout(self.proj(out))  # смешиваем все данные и приводим к n_embd
        return out


# FeedForward преобразует представление каждой позиции с помощью обычной нейросети.
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # увеличиваем размерность с n_embd до 4 * n_embd
            nn.ReLU(),                        # применяем нелинейность
            nn.Linear(4 * n_embd, n_embd),    # сжимаем обратно к n_embd
            nn.Dropout(dropout),              # рандомно выключаем часть нейронов
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    # residual connections: часть данных минует обработчик (x + f(x))
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        # Все размеры приходят СНАРУЖИ (из config), а не из глобалов файла.
        super().__init__()
        self.block_size = block_size  # запоминаем — generate() обрезает по нему

        # each token directly reads off the logits for the next token from a lookup table
        # token_embedding_table превращает каждый ID токена в обучаемый вектор размерности n_embd.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position_embedding_table даёт каждой позиции свой обучаемый вектор.
        # Это нужно, потому что один и тот же токен в начале последовательности
        # и ближе к концу должны обрабатываться с учётом их положения.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # blocks — n_layer Transformer-блоков, через которые проходят представления токенов.
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        # ln_f — финальная нормализация.
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        # lm_head преобразует каждый вектор размерности n_embd в набор оценок
        # для всех возможных следующих токенов.
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C): превращаем каждый токен в вектор n_embd
        # позиции 0..T-1, на том же устройстве что и idx
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C): значения сложились с позициями
        x = self.blocks(x)     # (B,T,C): проходит через n_layer блоков трансформера
        x = self.ln_f(x)       # (B,T,C): финальная нормализация
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # считаем потери по батчу

        return logits, loss

    def generate(self, idx, max_new_tokens, device, temperature=1.0, top_k=None):
        """
        Генерирует max_new_tokens новых токенов, продолжая последовательность idx.

        idx              — (B, T) текущий контекст (id токенов)
        max_new_tokens   — сколько токенов сгенерировать
        device           — "mps"/"cuda"/"cpu"
        temperature      — чем больше, тем разнообразнее (1.0 — обычный сэмплинг,
                           <1 — увереннее/детерминированнее, >1 — хаотичнее)
        top_k            — если задан, оставляем только k самых вероятных токенов
                           перед сэмплированием (обрезка "хвоста" распределения)
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] / temperature  # (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # optional: оставляем только top_k самых вероятных токенов
                if top_k is not None:
                    v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0  # всё ниже k-го — в ноль
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # пере-нормализуем
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return idx
