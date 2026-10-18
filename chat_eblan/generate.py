"""
Генерация текста обученной моделью Chat_eblan (минимальная версия M0).

Здесь два способа использовать модель:
1) свободная генерация (это main): даём начало текста, модель продолжает;
2) чат — импортирует из этого файла load_model()/load_tokenizer() (см. chat.py).

Запуск:  python generate.py "привет"        (из папки chat_eblan/)
         python generate.py                  (без аргумента — дефолтный промпт)
"""

import torch

import config
from model import GPTLanguageModel
from tokenizer import ChatTokenizer


def load_model(ckpt_path=config.CKPT_PATH, device=None):
    """
    Загружает обученную модель из чекпоинта.

    Возвращает (model, ckpt_config):
      model        — модель с восстановленными весами, уже в режиме eval()
      ckpt_config  — словарь гиперпараметров, сохранённый при обучении
                     (vocab_size, block_size, n_embd, n_head, n_layer, dropout)
    """
    device = device or config.DEVICE
    # weights_only=False: доверяем своему чекпоинту (в нём не только тензоры,
    # но и словарь с конфигом). map_location переносит веса на нужное устройство.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt['config']

    model = GPTLanguageModel(
        vocab_size=c['vocab_size'],
        block_size=c['block_size'],
        n_embd=c['n_embd'],
        n_head=c['n_head'],
        n_layer=c['n_layer'],
        dropout=c['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()  # выключаем dropout для инференса
    return model, c


def load_tokenizer(path=config.TOKENIZER_PATH):
    """Загружает обученный BPE-токенизатор из {path}.model."""
    tok = ChatTokenizer()
    tok.load(path)
    return tok


def main():
    import sys

    model, ckpt_config = load_model()
    tok = load_tokenizer()
    device = config.DEVICE

    # промпт: берём из аргумента командной строки или дефолтный
    prompt = sys.argv[1] if len(sys.argv) > 1 else f"{config.USER_TOKEN}привет{config.END_TOKEN}{config.ASSISTANT_TOKEN}"
    print(f"[generate] промпт: {prompt!r}")

    # кодируем промпт в токены и кладём в тензор формы (1, T)
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # генерируем продолжение
    out = model.generate(
        idx,
        max_new_tokens=64,
        device=device,
        temperature=0.8,  # <1 — увереннее, >1 — разнообразнее
        top_k=20,         # оставляем 20 самых вероятных токенов
    )

    # декодируем только СГЕНЕРИРОВАННУЮ часть (без исходного промпта)
    new_tokens = out[0, len(ids):].tolist()
    text = tok.decode(new_tokens)
    print(f"[generate] продолжение:\n{text}")


if __name__ == "__main__":
    main()
