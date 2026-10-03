"""Corpus-trained byte-level BPE with a frozen special-token contract."""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers


SPECIAL_TOKENS = ("[PAD]", "[BOS]", "[EOS]", "[UNK]")


class CaptionTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        if [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS] != list(range(4)):
            raise ValueError("Caption tokenizer special tokens must have IDs 0..3")
        if tokenizer.get_vocab_size() < 260:
            raise ValueError("Caption tokenizer must include a complete byte alphabet")
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    @classmethod
    def train(cls, captions, *, vocab_size: int, min_frequency: int):
        if vocab_size < 260 or min_frequency < 1:
            raise ValueError("Byte-level BPE needs vocab_size >= 260 and min_frequency >= 1")
        tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(captions, trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ))
        return cls(tokenizer)

    @classmethod
    def from_json(cls, text):
        return cls(Tokenizer.from_str(text))

    def to_json(self):
        return self.tokenizer.to_str()

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode_target(self, caption: str, *, max_length: int):
        encoded = self.tokenizer.encode(caption).ids
        if not encoded or any(i < 4 for i in encoded):
            raise ValueError("Caption must be nonempty and cannot contain special tokens")
        if len(encoded) + 1 > max_length:
            raise ValueError("Caption exceeds generation limit; no silent truncation")
        return [self.bos_id, *encoded, self.eos_id]

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)
