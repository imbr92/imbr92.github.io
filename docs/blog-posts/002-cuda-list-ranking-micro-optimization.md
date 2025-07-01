---
layout: post
title: "002 - CUDA List Ranking Micro-Optimization"
permalink: /blog-posts/002-cuda-list-ranking-micro-optimization
nav_exclude: true
---

I was recently involved in a discussion on how to do fast parallel list ranking in CUDA. There were some subtleties with the exact setup, but we can approximate it as follows:

You are given an integer `head` and two lists (already on device) `d_next` and `d_rank` each of length `n`. `d_next` describes an implicit linked list with head `head` and the element after any given element `e` is `d_next[e]`. The tail of the linked list points to `-1`. That is, `[head, d_next[head], d_next[d_next[head]], ..., -1]` forms a list of length `n + 1` containing `{-1, 0, ..., n - 1}` each exactly once (if `-1` was replaced with `head`, `d_next` would correspond exactly to an element of the symmetric group `S_n` forming a single `n`-cycle).

We would like to fill `d_rank` such that `d_rank[i]` is the index of element `i` in the linked list specified by `d_next`. Abusing notation slightly, `d_rank[i]` is defined such that

$$ d\_next^{d\_rank[i]}[head] = i$$

holds for all `i`. It is easy to see that this is well-defined when `d_next` satisfies the properties mentioned above.

Concretely, we can assume that we are tasked with completing a host function with the following signature:

```cpp
void list_rank(int32_t head, const int32_t *d_next, int32_t *d_rank, int32_t n);
```

For our particular set up, we care about `n` on the order of `1e8` and writing code capable of running on an eGPU RTX3070 (compute capability 8.6) compiled with NVCC using CUDA 12.8. There are many strategies for performing parallel list-ranking, but for this case study we'll focus on a modified version of the Helman-Jaja algorithm. In `list_rank`, we will perform the following `4` steps.

1. Copy Kernel: Copy `d_next` into an `int2* packed_next`, such that `packed_next[i].x = d_next[i]`
2. Sublist Relative Offset Kernel: Launch a kernel with `(k + 127) / 128` blocks and `128` threads per block. Define `S` to be a list of `k` unique elements of `{0, ..., n - 1}` such that `head` is in `S`. For each of the `k` threads, Pick a unique sublist head as a starting position. Iterate starting from the picked sublist head until we reach another sublist head (or `-1`), computing the relative offset of each element from the starting sublist head. Note that this effectively computes for each element, the distance to the nearest preceeding sublist head. Store this relative offset in `packed_next[i].y`. Also overwrite `packed_next[i].x` to be the index (in `S`) of the sublist head of element `i`. For each sublist head, also keep track of how large the sublist corresponding to that head is and the next sublist head. That is, for the `j`-th sublist head, store the size of the sublist starting at the `j`-th sublist head in `sublist_info[j].x` and the next sublist head in `sublist_info[j].y`.
3. On the CPU, compute a scan (prefix sum) of the sublist sizes in the ordering specified by `sublist_info[].y`.
4. Global Offset Kernel: For each `i`, do `d_rank[i] = sublist_info[packed_next[i].x].x + packed_next[i].y`

In step `3`, since we access `packed_next` in the CPU, we must either allocate it via `cudaMallocManaged`, `cudaMemcpy` it to and from the GPU, or do this entire step on the GPU. From my benchmarking, the first two of these options were roughly the same speed, and the third was slightly slower. This may not hold in a PCIe-connected GPU setup as opposed to the eGPU + Thunderbolt 3 setup I used. Also, in the above, `k` was a hyperparameter. For our problem size, the tradeoff involved in parameterizing `k` (for any reasonable value of `k`), was between increasing `k` in order to make step `2` run faster vs. decreasing it to make step `3` run faster. I hand optimized this to be ~200k.

We are also given the guarantee that the linked list which is given is generated uniformly at random from all possibly `n`-length linked lists, so for convenience, we choose `S = {0, ..., k - 2} U {head}`.

Running this through `nsys`, we get the following profile trace

![nsys output](/assets/images/post-002/nsys-output.webp)

From above, we can see that the steps take roughly 3 ms, 90 ms, 7 ms and 5 ms respectively, so the bulk of the time spent is on the second step - computing relative offsets for each index. Intuitively, this makes sense, since steps `1` and `4` can both make use of coalesced GMEM accesses, and though step `3` is done serially, the CPU is clocked quite a bit faster than the GPU (for my machine the CPU is clocked at 4.7 GHz while the GPU is only at 1.7 GHz) and more importantly, step `3` only does `O(k) << O(n)` work.

Based on this profiling distribution, we will focus on optimizing step `2` as this will likely yield the most bang for our buck. The corresponding kernel was as follows.

```cpp

template<const int NUM_LISTS>
__device__ inline bool is_sublist_head(const int32_t head, const int32_t cur_index){
    return (cur_index < NUM_LISTS - 1) || (cur_index == head) || cur_index == -1;
}

template <const int NUM_LISTS>
__global__ void compute_rel_offset(const int32_t head, int2* __restrict__ packed_next, int2* __restrict__ sublist_info, int32_t n){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int list_head = (tid == NUM_LISTS - 1) ? head : tid;

    int cur = list_head, rk = 0;
    if(tid < NUM_LISTS){
        do{
            int2 to = packed_next[cur];
            packed_next[cur] = {tid, rk++};
            cur = to.x;
        }
        while(!is_sublist_head<NUM_LISTS>(head, cur));
    }

    if(tid < NUM_LISTS){
        if(cur == head) cur = NUM_LISTS - 1;
        int2 out{cur, rk};
        sublist_info[tid] = out;
    }
}

...

// Later launched with
compute_rel_offset<k><<<k / 128, 128>>>(head, packed_next, sublist_info, n);
```

The bulk of the computation is done in the do-while loop above, reproduced below for convenience.

```cpp
        do{
            int2 to = packed_next[cur];
            packed_next[cur] = {tid, rk++};
            cur = to.x;
        }
        while(!is_sublist_head<NUM_LISTS>(head, cur));
```

In this loop, the very first access to `packed_next` is nicely coalesced - all accesses by threads in a given warp (other than possibly the very last one) for the first iteration are of consecutive elements by our choice of `S`. However, after this first iteration, there is no guarantee of such coalescing (in fact we expect very little coalescing). This kernel is also not very computationally intensive as it has an arithmetic intensity of `4/16 = 0.25 ops/byte` (of GMEM access). Thus, we'd expect to be bottlenecked by GMEM access bandwidth. There isn't too much we can do about the lack of coalesced accesses without changing our algorithmic approach.

We can confirm our observations above as well as get a better understanding of the overall kernel statistics by using the `ncu` output .

![ncu summary output](/assets/images/post-002/ncu-old-summary.webp)

The summary considerations for speedup are as expected. Quite low GMEM coalescing, long scoreboard stalls (waiting on the GMEM accesses), and a low L2 cache hit rate. When we look at the SASS with ARP samples, we see the following

![ncu sass](/assets/images/post-002/ncu-old-sass.webp)

The live arp stall sampling shows us primarily stalling on the following four instructions:

```asm
      LDG.E R0, [R2.64]                     ; 13.46%
      ...
      STG.E.64 [R2.64], R4                  ; 12.43%
      IMAD.MOV.U32 R5, RZ, RZ, R7           ; 18.00%
      ISETP.GT.AND P0, PT, R0, 0x2fffe, PT  ; 55.75%
```

Here, there are two dependencies
1. `ISETP.GT.AND` has a long scoreboard dependency on the 64-bit global load `LDG.E`
2. `IMAD.MOV.U32` has a long scoreboard dependency on the 64-bit global store `STG.E.64`

The stalls caused by `LDG.E` are necessary and the long scoreboard dependency makes sense, since we load into `R0` in the load and use this value in the `ISETP.GT.AND`. We must wait for the load before going to the next iteration because we must load from `packed_next[cur]` to find the index of the next element in the linked list.

On the other hand, the stalls on `STG.E.64` are different. We do not need to finish storing `{tid, rk + 1}` in `packed_next[cur]` before we go on to the next iteration. We are able to reason this because we know that `d_next` is given such that the data forms an `n`-cycle. This means that any element in `packed_next` will only be visited once in this kernel. However, the compiler does not know this, and thus must ensure that `packed_next[cur]` is updated so that if it is accessed in a later iteration, it is consistent with the post-store value. This introduces an unnecessary Read-After-Write (RAW) hazard in the code and limits the amount of instruction-level parallelism (ILP) able to be performed by the GPU.

As a side note, I'm not completely sure why the `IMAD.MOV.U32` instruction has a dependency on the store. From some quick reading online, it seems that it may have something to do with SASS serializing instructions after stores for memory ordering reasons (similar to why we must specify memory orderings for operations involving C++ atomics). I am also not entirely sure how to get `ncu-ui` to cleanly show the dependence of loads from one iteration on the stores from a previous iteration. I was able to see this by manually unrolling the loop, but I'd imagine there is a better way (I'd be very interested if anyone knows!)

We can fix this dependency by making the independence of these two explicit. Taking `d_next` as an argument to the kernel and switching the hot loop to the following,

```cpp
        do{
            int to = d_next[cur];
            packed_next[cur] = {tid, rk++};
            cur = to;
        }
        while(!is_sublist_head<NUM_LISTS>(head, cur));
```

Now, the load and stores are in different arrays, and with the `__restrict__` keyword, we know they are not aliasing overlapping memory regions. With this change, the SASS now looks like this

![ncu new sass](/assets/images/post-002/ncu-new-sass.webp)

This looks fairly similar, with the only change of note being the opcode change from `LDG.E.64` to `LDG.E.CONSTANT`. This data is not actually in constant memory, but it is using the read-only cache path. I don't believe this has too much of a performance impact since we have a highly irregular memory access pattern resulting in almost all of our memory accesses being cache misses, but this is still worth noting. However, comparing the kernels on their memory statistics we see a substantial improvement

![ncu new memory workload](/assets/images/post-002/ncu-new-memory-workload.webp)

Memory throughput is up 72%. This makes sense, since LDST pipelines are able to operate independently with loads and stores to GMEM. By making the independence of our load and store operations explicit, we are able to make use of this hardware independence by decoupling these processes entirely. This further boosts ILP since we are able to remove a dependence between the stores and loads which lets us dispatch more instructions in parallel to the same warp. Finally, taking a look at the nsys output for this new kernel, we see that the second step only takes 53 ms - a 42% reduction!

![nsys output new](/assets/images/post-002/nsys-output-new.webp)
