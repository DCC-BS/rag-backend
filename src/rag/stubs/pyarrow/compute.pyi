"""
This type stub file was generated by pyright.
"""

_OptionsClassDoc = ...

def cast(arr, target_type=..., safe=..., options=..., memory_pool=...):
    """
    Cast array values to another data type. Can also be invoked as an array
    instance method.

    Parameters
    ----------
    arr : Array-like
    target_type : DataType or str
        Type to cast to
    safe : bool, default True
        Check for overflows or other unsafe conversions
    options : CastOptions, default None
        Additional checks pass by CastOptions
    memory_pool : MemoryPool, optional
        memory pool to use for allocations during function execution.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pyarrow as pa
    >>> arr = pa.array([datetime(2010, 1, 1), datetime(2015, 1, 1)])
    >>> arr.type
    TimestampType(timestamp[us])

    You can use ``pyarrow.DataType`` objects to specify the target type:

    >>> cast(arr, pa.timestamp('ms'))
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]

    >>> cast(arr, pa.timestamp('ms')).type
    TimestampType(timestamp[ms])

    Alternatively, it is also supported to use the string aliases for these
    types:

    >>> arr.cast('timestamp[ms]')
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]
    >>> arr.cast('timestamp[ms]').type
    TimestampType(timestamp[ms])

    Returns
    -------
    casted : Array
        The cast result as a new Array
    """
    ...

def index(data, value, start=..., end=..., *, memory_pool=...):
    """
    Find the index of the first occurrence of a given value.

    Parameters
    ----------
    data : Array-like
    value : Scalar-like object
        The value to search for.
    start : int, optional
    end : int, optional
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    index : int
        the index, or -1 if not found

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["Lorem", "ipsum", "dolor", "sit", "Lorem", "ipsum"])
    >>> pc.index(arr, "ipsum")
    <pyarrow.Int64Scalar: 1>
    >>> pc.index(arr, "ipsum", start=2)
    <pyarrow.Int64Scalar: 5>
    >>> pc.index(arr, "amet")
    <pyarrow.Int64Scalar: -1>
    """
    ...

def take(data, indices, *, boundscheck=..., memory_pool=...):
    """
    Select values (or records) from array- or table-like data given integer
    selection indices.

    The result will be of the same type(s) as the input, with elements taken
    from the input array (or record batch / table fields) at the given
    indices. If an index is null then the corresponding value in the output
    will be null.

    Parameters
    ----------
    data : Array, ChunkedArray, RecordBatch, or Table
    indices : Array, ChunkedArray
        Must be of integer type
    boundscheck : boolean, default True
        Whether to boundscheck the indices. If False and there is an out of
        bounds index, will likely cause the process to crash.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : depends on inputs
        Selected values for the given indices

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> indices = pa.array([0, None, 4, 3])
    >>> arr.take(indices)
    <pyarrow.lib.StringArray object at ...>
    [
      "a",
      null,
      "e",
      null
    ]
    """
    ...

def fill_null(values, fill_value):
    """Replace each null element in values with a corresponding
    element from fill_value.

    If fill_value is scalar-like, then every null element in values
    will be replaced with fill_value. If fill_value is array-like,
    then the i-th element in values will be replaced with the i-th
    element in fill_value.

    The fill_value's type must be the same as that of values, or it
    must be able to be implicitly casted to the array's type.

    This is an alias for :func:`coalesce`.

    Parameters
    ----------
    values : Array, ChunkedArray, or Scalar-like object
        Each null element is replaced with the corresponding value
        from fill_value.
    fill_value : Array, ChunkedArray, or Scalar-like object
        If not same type as values, will attempt to cast.

    Returns
    -------
    result : depends on inputs
        Values with all null elements replaced

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array([1, 2, None, 3], type=pa.int8())
    >>> fill_value = pa.scalar(5, type=pa.int8())
    >>> arr.fill_null(fill_value)
    <pyarrow.lib.Int8Array object at ...>
    [
      1,
      2,
      5,
      3
    ]
    >>> arr = pa.array([1, 2, None, 4, None])
    >>> arr.fill_null(pa.array([10, 20, 30, 40, 50]))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      2,
      30,
      4,
      50
    ]
    """
    ...

def top_k_unstable(values, k, sort_keys=..., *, memory_pool=...):
    """
    Select the indices of the top-k ordered elements from array- or table-like
    data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get top indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array
        Indices of the top-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.top_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      5,
      4,
      2
    ]
    """
    ...

def bottom_k_unstable(values, k, sort_keys=..., *, memory_pool=...):
    """
    Select the indices of the bottom-k ordered elements from
    array- or table-like data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get bottom indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array of indices
        Indices of the bottom-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.bottom_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      0,
      1,
      2
    ]
    """
    ...

def random(n, *, initializer=..., options=..., memory_pool=...):
    """
    Generate numbers in the range [0, 1).

    Generated values are uniformly-distributed, double-precision
    in range [0, 1). Algorithm and seed can be changed via RandomOptions.

    Parameters
    ----------
    n : int
        Number of values to generate, must be greater than or equal to 0
    initializer : int or str
        How to initialize the underlying random generator.
        If an integer is given, it is used as a seed.
        If "system" is given, the random generator is initialized with
        a system-specific source of (hopefully true) randomness.
        Other values are invalid.
    options : pyarrow.compute.RandomOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """
    ...

def field(*name_or_index):
    """Reference a column of the dataset.

    Stores only the field's name. Type and other information is known only when
    the expression is bound to a dataset having an explicit scheme.

    Nested references are allowed by passing multiple names or a tuple of
    names. For example ``('foo', 'bar')`` references the field named "bar"
    inside the field named "foo".

    Parameters
    ----------
    *name_or_index : string, multiple strings, tuple or int
        The name or index of the (possibly nested) field the expression
        references to.

    Returns
    -------
    field_expr : Expression
        Reference to the given field

    Examples
    --------
    >>> import pyarrow.compute as pc
    >>> pc.field("a")
    <pyarrow.compute.Expression a>
    >>> pc.field(1)
    <pyarrow.compute.Expression FieldPath(1)>
    >>> pc.field(("a", "b"))
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    >>> pc.field("a", "b")
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    """
    ...

def scalar(value):
    """Expression representing a scalar value.

    Parameters
    ----------
    value : bool, int, float or string
        Python value of the scalar. Note that only a subset of types are
        currently supported.

    Returns
    -------
    scalar_expr : Expression
        An Expression representing the scalar value
    """
    ...
