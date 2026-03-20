import torch

SPECIAL_SYMBOLS = ["<PAD>", "<UNK>", "<SOS>", "\n"]


class LogVocab:

    def __init__(self, frequencies: dict = {}, max_size=-1, min_freq=0, str2int: dict = {}):
        self.max_size = max_size
        self.min_freq = min_freq
        self.str2int = dict()
        self.int2str = dict()

        if str2int:
            self.str2int = str2int
            self.int2str = dict((v, k) for k, v in str2int.items())
        else:
            self.int2str[0] = "<PAD>"
            self.int2str[1] = "<UNK>"
            self.int2str[2] = "<SOS>"

            self.str2int["<PAD>"] = 0
            self.str2int["<UNK>"] = 1
            self.str2int["<SOS>"] = 2

            for index, (k, v) in enumerate(
                    reversed(
                        sorted(
                            frequencies.items(),
                            key=lambda item: item[1])
                    )
            ):
                if v >= self.min_freq:
                    self.str2int[k] = index + 3
                    self.int2str[index + 3] = k
                if len(self.str2int) >= max_size > 0:
                    break

    def encode(self, logs: list | torch.Tensor):
        encoded = []
        for log in logs:
            if log in self.str2int:
                encoded.append(self.str2int[log])
            else:
                encoded.append(self.str2int["<UNK>"])
        return torch.tensor(encoded, dtype=torch.int32)

    def decode(self, indexes: list | torch.Tensor):
        decoded = []
        for index in indexes:
            index = int(index)
            if index in self.int2str:
                decoded.append(self.int2str[index])
            else:
                decoded.append(self.int2str[1])
        return decoded

    def __len__(self):
        return len(self.str2int)


class CharVocab:
    def __init__(self):
        self.str2int = {}

        for i, c in enumerate(SPECIAL_SYMBOLS):
            self.str2int[c] = i

        for i in range(32, 127):
            self.str2int[chr(i)] = i - 32 + len(SPECIAL_SYMBOLS)
        self.int2str = dict((v, k) for k, v in self.str2int.items())

    def encode(self, text):
        encoded = []
        for c in text:
            if c in self.str2int:
                encoded.append(self.str2int[c])
            else:
                encoded.append(self.str2int["<UNK>"])
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indexes):
        decoded = ''
        for index in indexes:
            index = int(index)
            if index in self.int2str:
                decoded += self.int2str[index]
            else:
                decoded += self.int2str[0]
        return decoded

    def __len__(self):
        return len(self.str2int)


def create_embedding_matrix(log_vocab: LogVocab | CharVocab, dim=384):
    return torch.nn.Embedding(len(log_vocab), dim, padding_idx=0)
