# pin_memory

pinned memory does not mean GPU memory but non-paged CPU memory. [This post](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
explains why we need to set `pin_memory=True`, which avoids one implicit copy `CPU-to-CPU`.

Additionally, with pinned memory tensors you can use `x.cuda(non_blocking=True)` to perform the copy _asynchronously_ with respect to host.


A very good explanation of `pin_memory` [link](https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader)
