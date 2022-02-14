from typing import Iterator, List

from torch.utils.data import Sampler

class BucketSampler(Sampler[List[int]]):

    NUM_BUFFERED_BATCHES = 100

    def __init__(self, inner_sampler:Sampler, batch_size:int) -> None:
        self.inner_sampler = inner_sampler
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        buffer:List[int] = []
        for idx in self.inner_sampler:
            buffer.append(idx)

            if len(buffer) == self.NUM_BUFFERED_BATCHES * self.batch_size:
                buffer.sort()
                for i in range(self.NUM_BUFFERED_BATCHES):
                    yield buffer[i*self.batch_size : (i+1)*self.batch_size]
                buffer = []
        
        num_batches = len(buffer) // self.batch_size
        if num_batches > 0:
            buffer.sort()
            for i in range(num_batches):
                yield buffer[i*self.batch_size : (i+1)*self.batch_size]

    def __len__(self) -> int:
        return len(self.inner_sampler) // self.batch_size