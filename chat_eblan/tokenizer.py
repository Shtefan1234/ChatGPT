"""
Обёртка над minbpe RegexTokenizer для проекта Chat_eblan.

Зачем нужна обёртка:
- прячет детали minbpe (спецтокены, allowed_special) за простым API;
- гарантирует, что спецтокены чата (<|user|>, <|assistant|>, <|end|>)
  всегда регистрируются с одинаковыми id при обучении и при загрузке;
- даёт удобные методы train / save / load / encode / decode.

Как это работает:
- токенизатор обучается BPE на тексте переписок;
- спецтокены чата регистрируются ОТДЕЛЬНО от BPE и получают id
  ПОСЛЕ всех обычных токенов (VOCAB_SIZE, VOCAB_SIZE+1, ...);
- encode() умеет распознавать спецтокены внутри текста и выдавать
  им их фиксированные id, а не разбивать на байты.
"""

from minbpe import RegexTokenizer

import config


class ChatTokenizer:
    """
    Токенизатор чата поверх minbpe RegexTokenizer.

    Поля:
      tokenizer — внутренний RegexTokenizer из minbpe
      special   — dict вида {"<|user|>": 512, "<|assistant|>": 513, "<|end|>": 514}
    """

    def __init__(self):
        self.tokenizer = RegexTokenizer()
        self.special = dict(config.SPECIAL_TOKENS)  # копия, чтобы не менять config

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def train(self, text: str, verbose: bool = True):
        """
        Обучает BPE на тексте и регистрирует спецтокены чата.

        text     — строка, на которой учим словарь (весь корпус переписок)
        verbose  — печатать ли прогресс мержей
        """
        # 1) обучаем BPE: делаем (VOCAB_SIZE - 256) мержей поверх базовых байтов
        self.tokenizer.train(text, config.VOCAB_SIZE, verbose=verbose)

        # 2) регистрируем спецтокены чата — минbpe выдаёт им отдельные id
        #    и НЕ разбивает их на части при encode()
        self.tokenizer.register_special_tokens(dict(self.special))

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------
    def save(self, prefix: str = config.TOKENIZER_PATH):
        """
        Сохраняет токенизатор в два файла: {prefix}.model и {prefix}.vocab.
        .model — рабочий (грузится через load), .vocab — для человека.
        """
        self.tokenizer.save(prefix)

    def load(self, prefix: str = config.TOKENIZER_PATH):
        """
        Загружает токенизатор из {prefix}.model.
        Спецтокены восстанавливаются автоматически (minbpe пишет их в .model).
        """
        self.tokenizer.load(prefix + ".model")
        # после load minbpe уже восстановил self.special_tokens,
        # но наша копия self.special обновляем для единообразия
        self.special = dict(self.tokenizer.special_tokens)

    # ------------------------------------------------------------------
    # Кодирование / декодирование
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list:
        """
        Строку -> список id токенов.
        allowed_special="all" означает: если в text встретился спецтокен
        вида <|user|>, выдать ему его id, а не разрезать на байты.
        """
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, ids) -> str:
        """Список id -> строка (спецтокены декодируются обратно в <|user|> и т.д.)."""
        return self.tokenizer.decode(list(ids))

    # ------------------------------------------------------------------
    # Полезные свойства
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Итоговый размер словаря = обычные токены + спецтокены."""
        return config.VOCAB_SIZE_TOTAL

    def token_id(self, token: str) -> int:
        """id конкретного спецтокена, например token_id("<|end|>")."""
        return self.special[token]


if __name__ == "__main__":
    # Быстрая самопроверка: обучаем на паре фраз и проверяем roundtrip.
    toy = "<|user|>привет<|end|><|assistant|>привет<|end|>"
    ct = ChatTokenizer()
    ct.train(toy, verbose=False)
    ids = ct.encode(toy)
    back = ct.decode(ids)
    print("encode:", ids)
    print("decode:", back)
    print("roundtrip OK:", back == toy)
    print("vocab_size:", ct.vocab_size)
