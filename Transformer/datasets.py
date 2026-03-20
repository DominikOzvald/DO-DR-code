import torch
from torch.utils.data import Dataset
from RNNVAE.embeddings import CharVocab
from data_utils import extract_raw
import os
import bisect
import torch.nn.functional as F


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
