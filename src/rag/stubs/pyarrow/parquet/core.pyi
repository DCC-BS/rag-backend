"""
This type stub file was generated by pyright.
"""

_DNF_filter_doc = ...

def filters_to_expression(filters):  # -> Any:
    """
    Check if filters are well-formed and convert to an ``Expression``.

    Parameters
    ----------
    filters : List[Tuple] or List[List[Tuple]]

    Notes
    -----
    See internal ``pyarrow._DNF_filter_doc`` attribute for more details.

    Examples
    --------

    >>> filters_to_expression([('foo', '==', 'bar')])
    <pyarrow.compute.Expression (foo == "bar")>

    Returns
    -------
    pyarrow.compute.Expression
        An Expression representing the filters
    """
    ...

_filters_to_expression = ...

class ParquetFile:
    """
    Reader interface for a single Parquet file.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
        Readable source. For passing bytes or buffer-like file containing a
        Parquet file, use pyarrow.BufferReader.
    metadata : FileMetaData, default None
        Use existing metadata object, rather than reading from file.
    common_metadata : FileMetaData, default None
        Will be used in reads for pandas schema metadata if not found in the
        main file's metadata, no other uses at the moment.
    read_dictionary : list
        List of column names to read directly as DictionaryArray.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    pre_buffer : bool, default False
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3). If True, Arrow will use a
        background I/O thread pool.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds.
    decryption_properties : FileDecryptionProperties, default None
        File decryption properties for Parquet Modular Encryption.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    page_checksum_verification : bool, default False
        If True, verify the checksum for each page read from the file.

    Examples
    --------

    Generate an example PyArrow Table and write it to Parquet file:

    >>> import pyarrow as pa
    >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    >>> import pyarrow.parquet as pq
    >>> pq.write_table(table, 'example.parquet')

    Create a ``ParquetFile`` object from the Parquet file:

    >>> parquet_file = pq.ParquetFile('example.parquet')

    Read the data:

    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Create a ParquetFile object with "animal" column as DictionaryArray:

    >>> parquet_file = pq.ParquetFile('example.parquet',
    ...                               read_dictionary=["animal"])
    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: dictionary<values=string, indices=int32, ordered=0>
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [  -- dictionary:
    ["Flamingo","Parrot",...,"Brittle stars","Centipede"]  -- indices:
    [0,1,2,3,4,5]]
    """
    def __init__(
        self,
        source,
        *,
        metadata=...,
        common_metadata=...,
        read_dictionary=...,
        memory_map=...,
        buffer_size=...,
        pre_buffer=...,
        coerce_int96_timestamp_unit=...,
        decryption_properties=...,
        thrift_string_size_limit=...,
        thrift_container_size_limit=...,
        filesystem=...,
        page_checksum_verification=...,
    ) -> None: ...
    def __enter__(self):  # -> Self:
        ...
    def __exit__(self, *args, **kwargs):  # -> None:
        ...
    @property
    def metadata(self):
        """
        Return the Parquet metadata.
        """
        ...

    @property
    def schema(self):
        """
        Return the Parquet schema, unconverted to Arrow types
        """
        ...

    @property
    def schema_arrow(self):
        """
        Return the inferred Arrow schema, converted from the whole Parquet
        file's schema

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read the Arrow schema:

        >>> parquet_file.schema_arrow
        n_legs: int64
        animal: string
        """
        ...

    @property
    def num_row_groups(self):
        """
        Return the number of row groups of the Parquet file.

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.num_row_groups
        1
        """
        ...

    def close(self, force: bool = ...):  # -> None:
        ...
    @property
    def closed(self) -> bool: ...
    def read_row_group(self, i, columns=..., use_threads=..., use_pandas_metadata=...):
        """
        Read a single row group from a Parquet file.

        Parameters
        ----------
        i : int
            Index of the individual row group that we want to read.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row group as a table (of columns)

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_group(0)
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,100]]
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
        ...

    def read_row_groups(self, row_groups, columns=..., use_threads=..., use_pandas_metadata=...):
        """
        Read a multiple row groups from a Parquet file.

        Parameters
        ----------
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row groups as a table (of columns).

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_groups([0,0])
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,...,2,4,4,5,100]]
        animal: [["Flamingo","Parrot","Dog",...,"Brittle stars","Centipede"]]
        """
        ...

    def iter_batches(self, batch_size=..., row_groups=..., columns=..., use_threads=..., use_pandas_metadata=...):
        """
        Read streaming batches from a Parquet file.

        Parameters
        ----------
        batch_size : int, default 64K
            Maximum number of records to yield per batch. Batches may be
            smaller if there aren't enough rows in the file.
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : boolean, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : boolean, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Yields
        ------
        pyarrow.RecordBatch
            Contents of each batch as a record batch

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')
        >>> for i in parquet_file.iter_batches():
        ...     print("RecordBatch")
        ...     print(i.to_pandas())
        ...
        RecordBatch
           n_legs         animal
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede
        """
        ...

    def read(self, columns=..., use_threads=..., use_pandas_metadata=...):
        """
        Read a Table from Parquet format.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read a Table:

        >>> parquet_file.read(columns=["animal"])
        pyarrow.Table
        animal: string
        ----
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
        ...

    def scan_contents(self, columns=..., batch_size=...):
        """
        Read contents of file for the given columns and batch size.

        Notes
        -----
        This function's primary purpose is benchmarking.
        The scan is executed on a single thread.

        Parameters
        ----------
        columns : list of integers, default None
            Select columns to read, if None scan all columns.
        batch_size : int, default 64K
            Number of rows to read at a time internally.

        Returns
        -------
        num_rows : int
            Number of rows in file

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.scan_contents()
        6
        """
        ...

_SPARK_DISALLOWED_CHARS = ...
_parquet_writer_arg_docs = ...
_parquet_writer_example_doc = ...

class ParquetWriter:
    __doc__ = ...
    def __init__(
        self,
        where,
        schema,
        filesystem=...,
        flavor=...,
        version=...,
        use_dictionary=...,
        compression=...,
        write_statistics=...,
        use_deprecated_int96_timestamps=...,
        compression_level=...,
        use_byte_stream_split=...,
        column_encoding=...,
        writer_engine_version=...,
        data_page_version=...,
        use_compliant_nested_type=...,
        encryption_properties=...,
        write_batch_size=...,
        dictionary_pagesize_limit=...,
        store_schema=...,
        write_page_index=...,
        write_page_checksum=...,
        sorting_columns=...,
        store_decimal_as_integer=...,
        **options,
    ) -> None: ...
    def __del__(self):  # -> None:
        ...
    def __enter__(self):  # -> Self:
        ...
    def __exit__(self, *args, **kwargs):  # -> Literal[False]:
        ...
    def write(self, table_or_batch, row_group_size=...):  # -> None:
        """
        Write RecordBatch or Table to the Parquet file.

        Parameters
        ----------
        table_or_batch : {RecordBatch, Table}
        row_group_size : int, default None
            Maximum number of rows in each written row group. If None,
            the row group size will be the minimum of the input
            table or batch length and 1024 * 1024.
        """
        ...

    def write_batch(self, batch, row_group_size=...):  # -> None:
        """
        Write RecordBatch to the Parquet file.

        Parameters
        ----------
        batch : RecordBatch
        row_group_size : int, default None
            Maximum number of rows in written row group. If None, the
            row group size will be the minimum of the RecordBatch
            size and 1024 * 1024.  If set larger than 64Mi then 64Mi
            will be used instead.
        """
        ...

    def write_table(self, table, row_group_size=...):  # -> None:
        """
        Write Table to the Parquet file.

        Parameters
        ----------
        table : Table
        row_group_size : int, default None
            Maximum number of rows in each written row group. If None,
            the row group size will be the minimum of the Table size
            and 1024 * 1024.  If set larger than 64Mi then 64Mi will
            be used instead.

        """
        ...

    def close(self):  # -> None:
        """
        Close the connection to the Parquet file.
        """
        ...

    def add_key_value_metadata(self, key_value_metadata):  # -> None:
        """
        Add key-value metadata to the file.
        This will overwrite any existing metadata with the same key.

        Parameters
        ----------
        key_value_metadata : dict
            Keys and values must be string-like / coercible to bytes.
        """
        ...

EXCLUDED_PARQUET_PATHS = ...
_read_docstring_common = ...
_parquet_dataset_example = ...

class ParquetDataset:
    __doc__ = ...
    def __init__(
        self,
        path_or_paths,
        filesystem=...,
        schema=...,
        *,
        filters=...,
        read_dictionary=...,
        memory_map=...,
        buffer_size=...,
        partitioning=...,
        ignore_prefixes=...,
        pre_buffer=...,
        coerce_int96_timestamp_unit=...,
        decryption_properties=...,
        thrift_string_size_limit=...,
        thrift_container_size_limit=...,
        page_checksum_verification=...,
        use_legacy_dataset=...,
    ) -> None: ...
    def equals(self, other): ...
    def __eq__(self, other) -> bool: ...
    @property
    def schema(self):
        """
        Schema of the Dataset.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_schema',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_schema/')

        Read the schema:

        >>> dataset.schema
        n_legs: int64
        animal: string
        year: dictionary<values=int32, indices=int32, ordered=0>
        """
        ...

    def read(self, columns=..., use_threads=..., use_pandas_metadata=...):
        """
        Read (multiple) Parquet files as a single pyarrow.Table.

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the dataset. The partition fields
            are not automatically included.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_read',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_read/')

        Read the dataset:

        >>> dataset.read(columns=["n_legs"])
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[5],[2],[4,100],[2,4]]
        """
        ...

    def read_pandas(self, **kwargs):
        """
        Read dataset including pandas metadata, if any. Other arguments passed
        through to :func:`read`, see docstring for further details.

        Parameters
        ----------
        **kwargs : optional
            Additional options for :func:`read`

        Examples
        --------
        Generate an example parquet file:

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                    'n_legs': [2, 2, 4, 4, 5, 100],
        ...                    'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                    "Brittle stars", "Centipede"]})
        >>> table = pa.Table.from_pandas(df)
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'table_V2.parquet')
        >>> dataset = pq.ParquetDataset('table_V2.parquet')

        Read the dataset with pandas metadata:

        >>> dataset.read_pandas(columns=["n_legs"])
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[2,2,4,4,5,100]]

        >>> dataset.read_pandas(columns=["n_legs"]).schema.pandas_metadata
        {'index_columns': [{'kind': 'range', 'name': None, 'start': 0, ...}
        """
        ...

    @property
    def fragments(self):  # -> list[Any]:
        """
        A list of the Dataset source fragments or pieces with absolute
        file paths.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_fragments',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_fragments/')

        List the fragments:

        >>> dataset.fragments
        [<pyarrow.dataset.ParquetFileFragment path=dataset_v2_fragments/...
        """
        ...

    @property
    def files(self):
        """
        A list of absolute Parquet file paths in the Dataset source.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path='dataset_v2_files',
        ...                     partition_cols=['year'])
        >>> dataset = pq.ParquetDataset('dataset_v2_files/')

        List the files:

        >>> dataset.files
        ['dataset_v2_files/year=2019/...-0.parquet', ...
        """
        ...

    @property
    def filesystem(self):
        """
        The filesystem type of the Dataset source.
        """
        ...

    @property
    def partitioning(self):
        """
        The partitioning of the Dataset source, if discovered.
        """
        ...

_read_table_docstring = ...
_read_table_example = ...

def read_table(
    source,
    *,
    columns=...,
    use_threads=...,
    schema=...,
    use_pandas_metadata=...,
    read_dictionary=...,
    memory_map=...,
    buffer_size=...,
    partitioning=...,
    filesystem=...,
    filters=...,
    use_legacy_dataset=...,
    ignore_prefixes=...,
    pre_buffer=...,
    coerce_int96_timestamp_unit=...,
    decryption_properties=...,
    thrift_string_size_limit=...,
    thrift_container_size_limit=...,
    page_checksum_verification=...,
): ...
def read_pandas(source, columns=..., **kwargs): ...
def write_table(
    table,
    where,
    row_group_size=...,
    version=...,
    use_dictionary=...,
    compression=...,
    write_statistics=...,
    use_deprecated_int96_timestamps=...,
    coerce_timestamps=...,
    allow_truncated_timestamps=...,
    data_page_size=...,
    flavor=...,
    filesystem=...,
    compression_level=...,
    use_byte_stream_split=...,
    column_encoding=...,
    data_page_version=...,
    use_compliant_nested_type=...,
    encryption_properties=...,
    write_batch_size=...,
    dictionary_pagesize_limit=...,
    store_schema=...,
    write_page_index=...,
    write_page_checksum=...,
    sorting_columns=...,
    store_decimal_as_integer=...,
    **kwargs,
):  # -> None:
    ...

_write_table_example = ...

def write_to_dataset(
    table,
    root_path,
    partition_cols=...,
    filesystem=...,
    use_legacy_dataset=...,
    schema=...,
    partitioning=...,
    basename_template=...,
    use_threads=...,
    file_visitor=...,
    existing_data_behavior=...,
    **kwargs,
):  # -> None:
    """Wrapper around dataset.write_dataset for writing a Table to
    Parquet format by partitions.
    For each combination of partition columns and values,
    a subdirectories are created in the following
    manner:

    root_dir/
      group1=value1
        group2=value1
          <uuid>.parquet
        group2=value2
          <uuid>.parquet
      group1=valueN
        group2=value1
          <uuid>.parquet
        group2=valueN
          <uuid>.parquet

    Parameters
    ----------
    table : pyarrow.Table
    root_path : str, pathlib.Path
        The root directory of the dataset.
    partition_cols : list,
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    use_legacy_dataset : bool, optional
        Deprecated and has no effect from PyArrow version 15.0.0.
    schema : Schema, optional
        This Schema of the dataset.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the
        ``pyarrow.dataset.partitioning()`` function or a list of field names.
        When providing a list of field names, you can use
        ``partitioning_flavor`` to drive which partitioning type should be
        used.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to "guid-{i}.parquet".
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores.
    file_visitor : function
        If set, this function will be called with a WrittenFile instance
        for each file created during the call.  This object will have both
        a path attribute and a metadata attribute.

        The path attribute will be a string containing the path to
        the created file.

        The metadata attribute will be the parquet metadata of the file.
        This metadata will have the file path attribute set and can be used
        to build a _metadata file.  The metadata attribute will be None if
        the format is not parquet.

        Example visitor which simple collects the filenames created::

            visited_paths = []

            def file_visitor(written_file):
                visited_paths.append(written_file.path)

    existing_data_behavior : 'overwrite_or_ignore' | 'error' | \
'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination. The default behaviour is 'overwrite_or_ignore'.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'error' will raise an error if any data exists in the destination.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    **kwargs : dict,
        Used as additional kwargs for :func:`pyarrow.dataset.write_dataset`
        function for matching kwargs, and remainder to
        :func:`pyarrow.dataset.ParquetFileFormat.make_write_options`.
        See the docstring of :func:`write_table` and
        :func:`pyarrow.dataset.write_dataset` for the available options.
        Using `metadata_collector` in kwargs allows one to collect the
        file metadata instances of dataset pieces. The file paths in the
        ColumnChunkMetaData will be set relative to `root_path`.

    Examples
    --------
    Generate an example PyArrow Table:

    >>> import pyarrow as pa
    >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
    ...                   'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    and write it to a partitioned dataset:

    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, root_path='dataset_name_3',
    ...                     partition_cols=['year'])
    >>> pq.ParquetDataset('dataset_name_3').files
    ['dataset_name_3/year=2019/...-0.parquet', ...

    Write a single Parquet file into the root folder:

    >>> pq.write_to_dataset(table, root_path='dataset_name_4')
    >>> pq.ParquetDataset('dataset_name_4/').files
    ['dataset_name_4/...-0.parquet']
    """
    ...

def write_metadata(schema, where, metadata_collector=..., filesystem=..., **kwargs):  # -> None:
    """
    Write metadata-only Parquet file from schema. This can be used with
    `write_to_dataset` to generate `_common_metadata` and `_metadata` sidecar
    files.

    Parameters
    ----------
    schema : pyarrow.Schema
    where : string or pyarrow.NativeFile
    metadata_collector : list
        where to collect metadata information.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred from `where` if path-like, else
        `where` is already a file-like object so no filesystem is needed.
    **kwargs : dict,
        Additional kwargs for ParquetWriter class. See docstring for
        `ParquetWriter` for more information.

    Examples
    --------
    Generate example data:

    >>> import pyarrow as pa
    >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    Write a dataset and collect metadata information.

    >>> metadata_collector = []
    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(
    ...     table, 'dataset_metadata',
    ...      metadata_collector=metadata_collector)

    Write the `_common_metadata` parquet file without row groups statistics.

    >>> pq.write_metadata(
    ...     table.schema, 'dataset_metadata/_common_metadata')

    Write the `_metadata` parquet file with row groups statistics.

    >>> pq.write_metadata(
    ...     table.schema, 'dataset_metadata/_metadata',
    ...     metadata_collector=metadata_collector)
    """
    ...

def read_metadata(where, memory_map=..., decryption_properties=..., filesystem=...):
    """
    Read FileMetaData from footer of a single Parquet file.

    Parameters
    ----------
    where : str (file path) or file-like object
    memory_map : bool, default False
        Create memory map when the source is a file path.
    decryption_properties : FileDecryptionProperties, default None
        Decryption properties for reading encrypted Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.

    Returns
    -------
    metadata : FileMetaData
        The metadata of the Parquet file

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({'n_legs': [4, 5, 100],
    ...                   'animal': ["Dog", "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, 'example.parquet')

    >>> pq.read_metadata('example.parquet')
    <pyarrow._parquet.FileMetaData object at ...>
      created_by: parquet-cpp-arrow version ...
      num_columns: 2
      num_rows: 3
      num_row_groups: 1
      format_version: 2.6
      serialized_size: ...
    """
    ...

def read_schema(where, memory_map=..., decryption_properties=..., filesystem=...):
    """
    Read effective Arrow schema from Parquet file metadata.

    Parameters
    ----------
    where : str (file path) or file-like object
    memory_map : bool, default False
        Create memory map when the source is a file path.
    decryption_properties : FileDecryptionProperties, default None
        Decryption properties for reading encrypted Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.

    Returns
    -------
    schema : pyarrow.Schema
        The schema of the Parquet file

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({'n_legs': [4, 5, 100],
    ...                   'animal': ["Dog", "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, 'example.parquet')

    >>> pq.read_schema('example.parquet')
    n_legs: int64
    animal: string
    """
    ...

__all__ = (
    "ColumnChunkMetaData",
    "ColumnSchema",
    "FileDecryptionProperties",
    "FileEncryptionProperties",
    "FileMetaData",
    "ParquetDataset",
    "ParquetFile",
    "ParquetLogicalType",
    "ParquetReader",
    "ParquetSchema",
    "ParquetWriter",
    "RowGroupMetaData",
    "SortingColumn",
    "Statistics",
    "read_metadata",
    "read_pandas",
    "read_schema",
    "read_table",
    "write_metadata",
    "write_table",
    "write_to_dataset",
    "_filters_to_expression",
    "filters_to_expression",
)
