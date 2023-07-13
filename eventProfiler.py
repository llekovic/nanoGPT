from contextlib import contextmanager
import resource
import gc
import torch


class EventProfiler:
    def __init__(self, device):
        self.device = device
        # Warm-up GPU
        torch.randn(3, 3, device=device) @ torch.randn(3, 3, device=device)
        torch.cuda.empty_cache()
        gc.collect()
        self.reset()

    def reset(self):
        """Reset the timer"""
        self.initialized_keys = set()
        self.time_data = dict()  # the time for each occurence of each event
        self.cuda_max_mem_data = dict()
        self.cuda_allocated_mem_data = dict()
        self.ram_allocated_mem_data = dict()

    def create_label_if_not_exists(self, label):
        # Update first and last occurrence of this label
        if label not in self.initialized_keys:
            self.time_data[label] = []
            self.cuda_max_mem_data[label] = []
            self.cuda_allocated_mem_data[label] = []
            self.ram_allocated_mem_data[label] = []
            self.initialized_keys.add(label)

    @contextmanager
    def __call__(self, label):
        torch.cuda.current_stream().synchronize()  # Wait for everything before me to finish

        # Measure the time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        yield
        torch.cuda.current_stream().synchronize()  # Wait for operations that happen during yield to finish
        end.record()

        torch.cuda.current_stream().synchronize()  # Need to wait once more for operations to finish
        self.create_label_if_not_exists(label)

        self.time_data[label].append(start.elapsed_time(end) / 1000)  # seconds

        self.cuda_max_mem_data[label].append(
            torch.cuda.max_memory_allocated() / (1024**3)
        )  # GiB
        self.cuda_allocated_mem_data[label].append(
            torch.cuda.memory_allocated() / (1024**3)
        )

        # **2 here, since resource.getrusage returns KiB
        self.ram_allocated_mem_data[label].append(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        )

        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def summary(self):
        return {
            "time": {k: torch.tensor(v) for k, v in self.time_data.items()},
            "cuda-max (GiB)": {
                k: torch.tensor(v) for k, v in self.cuda_max_mem_data.items()
            },
            "cuda-current (GiB)": {
                k: torch.tensor(v) for k, v in self.cuda_allocated_mem_data.items()
            },
            "ram (GiB)": {
                k: torch.tensor(v) for k, v in self.ram_allocated_mem_data.items()
            },
        }

    def save_results(self, addr):
        ret = self.summary()
        torch.save(ret, addr)
