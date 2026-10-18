from .base import Tokenizer
from .basic import BasicTokenizer
from .regex import RegexTokenizer

# GPT4Tokenizer требует установленного tiktoken (нужен интернет/докачка весов).
# Нам он не нужен — используем RegexTokenizer, поэтому делаем импорт опциональным,
# чтобы пакет работал без зависимости tiktoken.
try:
    from .gpt4 import GPT4Tokenizer
except ImportError:
    GPT4Tokenizer = None
