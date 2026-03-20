from torch.utils.data import Dataset
from Drain import LogParser
from data_utils import extract_with_parse, extract_raw, form_instances, count_logs, pad_frame_collate_fn
from embeddings import LogVocab, CharVocab
import os

PARSE_IN_DIR = "./"
PARSE_OUT_DIR = "./"

DEPTH = 4
ST = 0.2
MAX_CHILD = 100

MAX_SIZE = -1
MIN_FREQ = 0

DEFAULT_DICT = {
    "maxChild": MAX_CHILD,
    "st": ST,
    "rex": [],
    "depth": DEPTH,
    "maxSize": MAX_SIZE,
    "minFreq": MIN_FREQ,
}


def set_kwargs(arg_dict):
    for k in DEFAULT_DICT:
        if k not in arg_dict:
            arg_dict[k] = DEFAULT_DICT[k]


class LogDataSet(Dataset):
    def __init__(self, log_dir: str, log_vocab: LogVocab = None, log_format='<DateTime> <Content>', step=1,
                 frame_size=1, **kwargs):

        self.data = []
        self.vocab = log_vocab
        total_frequencies = {}

        set_kwargs(kwargs)

        parser = LogParser(log_format, PARSE_IN_DIR, PARSE_OUT_DIR, maxChild=kwargs["maxChild"], st=kwargs["st"],
                           rex=kwargs["rex"], depth=kwargs["depth"],
                           verbose=False)

        log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
        for log_file in log_files:
            frequencies, logs = extract_with_parse(os.path.join(log_dir, log_file), parser)
            self.data += form_instances(logs, step, frame_size)
            if log_vocab is None:
                count_logs(total_frequencies, frequencies)

        if self.vocab is None:
            self.vocab = LogVocab(total_frequencies, max_size=kwargs["maxSize"], min_freq=kwargs["minFreq"])

    def __getitem__(self, item):
        log = self.data[item]
        return self.vocab.encode(log)

    def __len__(self):
        return len(self.data)


class LogCharDataSet(Dataset):

    def __init__(self, log_dir: str, step=1, frame_size=1, cut_off=200, join_frame=True):
        super().__init__()
        self.data = []
        self.vocab = CharVocab()
        self.join_frame = join_frame
        log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
        for log_file in log_files:
            logs = extract_raw(os.path.join(log_dir, log_file))
            for i in range(0, len(logs), step):
                frame = logs[i:i + frame_size]
                frame = [log[:cut_off] for log in frame]
                if self.join_frame:
                    instance = "".join(frame)
                    if len(instance) > 1 and instance not in self.data:
                        self.data.append(instance)
                else:
                    self.data.append(frame)

    def _encode_frame(self,frame: str | list):
        if type(frame) is str:
            return self.vocab.encode(frame)
        else:
            return [self.vocab.encode(line) for line in frame]

    def __getitem__(self, item: int | slice):
        logs = self.data[item]
        if type(item) is int:
            return self._encode_frame(logs)
        else:
            return [self._encode_frame(frame) for frame in logs]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    step_size = 15
    frame_size = 15
    data_set = LogCharDataSet("../test_data", frame_size=frame_size, step=step_size,join_frame=False)

    x = [data_set[0],data_set[1][:10]]
    x,lens = pad_frame_collate_fn(x)
    print(x.shape)
#     re = [r"^[\.s]+$"]
#     data_set = LogDataSet("../test_data", minFreq=2, step=step_size, frame_size=frame_size, rex=re)
#     loader = DataLoader(dataset=data_set, shuffle=False, batch_size=2, collate_fn=fixed_pad_fn_factory(frame_size))
#     matrix = create_embedding_matrix(data_set.vocab,dim=5)
#     print(data_set.vocab.str2int)
#     batch = matrix(next(iter(loader)))
