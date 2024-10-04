# Performance Tuning Guidelines

The PyTorch data loader is designed to read from very large SOMA datasets, which are impractical to serve directly from in-core data structures. This document contains suggestions which may be helpful in improving the data loading performance.

The "best" configuration for any given model will depend on a number of variables (e.g., complexity of the model, available compute resources, etc). This document provides general guidelines only.

## Loader Configuration

General guidelines, approximately ordered by magnitude of impact:

* Read-ahead, i.e., `use_eager_fetch=True`, will improve throughput in any given epic, as the loader will attempt to always have an I/O buffer available. This requires additional memory (in the worst case, it will approximately double memory use). Read-ahead is enabled by default.
* When using the PyTorch DataLoader, there is normally benefit from multiple workers per GPU (controlled with the DataLoader `num_workers` parameter). In most environments, 1-2 workers per GPU is a good starting point, and exceeding 2 may decrease performance due to over-subscription of CPU and RAM on the host.
* If using shuffling (enabled by default): increasing the `shuffle_chunk_size` (default value 64) will increase read performance and decrease randomness.
* The `tiledbsoma` I/O buffer size, set by the `SOMATileDBContext` parameter `soma.init_buffer_bytes`, will decrease performance if it becomes too small.  The default is 1GiB and should be sufficient for most workloads. Experimenting with larger values (e.g., 2GiB) may show modest improvements in throughput.
* The mini-batch size (controlled with `batch_size`) will have a modest detrimental impact on performance as it becomes smaller, primarily due to the additional CPU overhead of creating more batches.
* The IO batch size (set with `io_batch_size`) will have a detrimental impact on performance if set too small. The default value (65536) is a reasonable starting point, and large increases have not shown any great benefit.

## Data Schema

TileDB and TileDB-SOMA provide control over the physical data layout, and specifying a layout tuned for your query pattern can improve performance for that specific workload.  The default SOMA data layout is intended to achieve reasonable performance for a wide variety of access patterns, and is not optimal for use with the data loader.

In most SOMA Experiments, the primary performance bottleneck is reading from the ``X`` matrix. The following guidelines are designed to reduce unutilized I/O:

* On the ``obs`` (first) dimension, specify space tile extents matching your expected shuffle chunk size.
* On the ``var`` (second) dimension, specify space tile extents matching your expected axis slice.
* Specify a tile capacity which is sufficiently large to achieve good compression in the filter pipeline, but not so large as to increase unnecessary I/O.  A good starting point is 64K-128K (the default is currently 100,000).

For example:

* If you typically use a shuffle chunk of 32, set the first dimension space tile to 16 or 32.
* If you typically read most ``var`` values, set the second dimension space tile to a large value (e.g., 4096 or 8192). If you always read _all_ ``var`` columns you could even set the tile size of the entire dimension extent.

In addition, you may find that increasing the compression level in the filter pipeline may provide a modest improvement in read throughput (at the cost of write throughput when creating the array).

```python
tiledbsoma.SparseNDArray.create(
    array_uri,
    type=array_data_type,   # e.g., pyarrow.float32()
    shape=array_shape,      # e.g., (48_000_000, 60_530)
    platform_config = {     # Tuning the array schema
        "tiledb": {
            "create": {
                "capacity": 2**16,
                "dims": {
                    "soma_dim_0": { "tile": 16 },
                    "soma_dim_1": { "tile": 2048 },
                }
            }
        }
    },
    context=a_soma_tiledb_context
)

```

For general information on performance tuning, refer to general background on [TileDB performance tuning](https://docs.tiledb.com/main/how-to/performance/performance-tips).

## Data Consolidation

TileDB arrays are commonly comprised of multiple array fragments, representing separate writes to the array. When the fragment count is high, or fragments contain interleaved coordinates, read performance can suffer. Consolidating and vacuuming the array may help performance. For more information, see the relevant [TileDB documentation](https://docs.tiledb.com/main/how-to/arrays/writing-arrays/consolidation-and-vacuuming).

## Host and Data Storage Configuration

General guidelines:

* If data is resident on network storage (e.g., AWS EBS or S3), ensure ample network bandwidth. For example, a `n` instance type (e.g., ``g4dn``) will typically provide significantly more throughput.
* Data loading requires sufficient CPU & RAM for the TileDB engine. While requirements vary by dataset, suggest at least 8 CPUs and 16-24 GiB of RAM per host GPU.
* On AWS, benchmarking has shown that loading of S3-resident data is approximately as performant as EBS-resident data. Using a faster storage system (e.g., local nvRAM ephemeral storage) may increase throughput somewhat, at the cost of additional data logistics.
