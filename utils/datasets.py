import torch
from torch.utils.data import Dataset
from Drain import LogParser
from utils.embeddings import CharVocab, LogVocab
from utils.data import extract_raw, extract_with_parse, form_instances, count_logs
import os
import bisect
import torch.nn.functional as F

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


class TransformerDataset(Dataset):
    def __init__(self, log_dir: str, step=1, frame_size=1, max_len=200, ):
        super().__init__()
        self.data = []
        self.vocab = CharVocab()
        self.step = step
        self.frame_size = frame_size
        self.file_starts = []
        self.max_len = max_len
        self.file_frames = []
        log_files = [file for file in os.listdir(log_dir) if file[-4:] == ".txt"]
        total_num_frames = 0
        for log_file in log_files:
            self.file_frames.append(total_num_frames)
            self.file_starts.append(len(self.data))
            logs = extract_raw(os.path.join(log_dir, log_file))
            self.data += [log[:max_len] for log in logs]
            num_frames = len(logs) // self.step
            if len(logs) % self.step:
                num_frames += 1

            total_num_frames += num_frames
        self.length = total_num_frames

    def _frame_ends(self, item):
        frame_start_file = bisect.bisect_right(self.file_frames, item) - 1

        frame_start = self.file_starts[frame_start_file] + (item - self.file_frames[frame_start_file]) * self.step
        frame_end = frame_start + self.frame_size
        frame_end_file = bisect.bisect_right(self.file_starts, frame_end) - 1
        if frame_start_file != frame_end_file:
            frame_end = self.file_starts[frame_start_file + 1]
        return frame_start, frame_end

    def __getitem__(self, item):
        frame_stat, frame_end = self._frame_ends(item)
        logs = self.data[frame_stat:frame_end]
        enc_logs = [self.vocab.encode(log) for log in logs]
        lengths = torch.Tensor([len(log) for log in enc_logs])
        frame = torch.stack([F.pad(log, (0, self.max_len - log.size(0)), value=0) for log in enc_logs]).to(torch.long)
        frame_len = frame.size(0)
        if frame_len < self.frame_size:
            frame = F.pad(frame, (0, 0, 0, self.frame_size - frame_len), value=0)
            lengths = torch.cat([lengths, torch.ones(self.frame_size - frame_len)])
            mask = torch.cat([torch.zeros(frame_len), torch.ones(self.frame_size - frame_len)])
        else:
            mask = torch.zeros(frame_len)
        return frame, lengths, mask

    def __len__(self):
        return self.length


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

    def _encode_frame(self, frame: str | list):
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
