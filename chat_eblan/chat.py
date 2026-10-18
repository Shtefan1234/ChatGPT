"""
Интерактивный чат с моделью Chat_eblan (минимальная версия M0).

Идея: держим "историю" диалога строкой в формате чата. На каждый ход
юзера мы:
1) дописываем в конец истории  <|user|>{сообщение}<|end|><|assistant|>;
2) кодируем историю в токены и обрезаем с начала по BLOCK_SIZE;
3) генерируем токены по одному, пока модель не выдаст <|end|>;
4) декодируем сгенерированное, печатаем ответ;
5) дописываем ответ в историю и повторяем.

Запуск:  python chat.py   (из папки chat_eblan/). Выход: quit / exit / Ctrl+C
"""

import torch
from torch.nn import functional as F

import config
from generate import load_model, load_tokenizer


def sample_next(logits, temperature=0.8, top_k=20):
    """
    Из логитов последнего токена сэмплирует следующий токен.

    logits     — (vocab_size,) оценки модели для следующего токена
    temperature— чем больше, тем разнообразнее (0.8 — чуть увереннее)
    top_k      — оставляем только k самых вероятных токенов
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if top_k is not None:
        v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
        probs[probs < v[-1]] = 0  # всё ниже k-го — в ноль
        probs = probs / probs.sum()  # пере-нормализуем
    # берём один токен по распределению вероятностей
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_reply(model, tok, prompt_ids, device, max_new_tokens=64):
    """
    Генерирует ответ ассистента, начиная с prompt_ids (id токенов промпта).

    Стоп-условие: модель сгенерировала <|end|> (конец реплики).
    Возвращает строку ответа (без <|end|>).
    """
    end_id = tok.token_id(config.END_TOKEN)  # id токена <|end|>
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        # обрезаем контекст по block_size (модель "видит" только последнее окно)
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        # берём логиты последнего токена и сэмплируем следующий
        next_id = sample_next(logits[0, -1, :]).item()
        # если модель поставила конец реплики — стоп
        if next_id == end_id:
            break
        generated.append(next_id)
        # дописываем новый токен в последовательность
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)

    # декодируем только что сгенерированные токены в текст
    return tok.decode(generated)


def trim_history(history_ids, block_size):
    """
    Обрезает историю токенов СЛЕВА, чтобы она помещалась в block_size.
    Нам нужны последние реплики контекста, старые — отбрасываем.
    """
    if len(history_ids) > block_size:
        return history_ids[-block_size:]
    return history_ids


def main():
    model, _ = load_model()
    tok = load_tokenizer()
    device = config.DEVICE

    # история — список id токенов всего, что сказано в чате (юзер + бот)
    history_ids = []

    print("=== Chat_eblan (минимальная версия M0) ===")
    print("Выход: quit / exit / Ctrl+C\n")

    while True:
        try:
            user_text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if user_text.lower() in ("quit", "exit"):
            print("Пока!")
            break
        if not user_text:
            continue

        # 1) дописываем реплику юзера в историю
        turn = f"{config.USER_TOKEN}{user_text}{config.END_TOKEN}{config.ASSISTANT_TOKEN}"
        history_ids.extend(tok.encode(turn))

        # 2) обрезаем историю по контексту и подаём модели
        context_ids = trim_history(history_ids, config.BLOCK_SIZE)

        # 3) генерируем ответ до <|end|>
        reply = generate_reply(model, tok, context_ids, device)

        # 4) дописываем ответ в историю (закрываем <|end|>)
        history_ids.extend(tok.encode(reply + config.END_TOKEN))
        # снова подрезаем, чтобы история не росла бесконечно
        history_ids = trim_history(history_ids, config.BLOCK_SIZE)

        print(f"Бот: {reply}\n")


if __name__ == "__main__":
    main()
