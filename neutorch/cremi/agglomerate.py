from .replace_values_inplace import replace_values_inplace
import daisy
import logging
import numpy as np
import waterz

logger = logging.getLogger(__name__)


def parallel_aff_agglomerate(
        affs,
        fragments,
        rag_provider,
        block_size,
        context,
        merge_function,
        threshold,
        num_workers):
    '''Agglomerate fragments in parallel using ``waterz``.

    Args:

        affs (`class:daisy.Array`):

            An array containing affinities.

        fragments (`class:daisy.Array`):

            An array containing fragments.

        rag_provider (`class:SharedRagProvider`):

            A RAG provider to read nodes from and write found edges to.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel, in world units.

        context (``tuple`` of ``int``):

            The context to consider for agglomeration, in world units.

        merge_function (``string``):

            The merge function to use for ``waterz``.

        threshold (``float``):

            Until which threshold to agglomerate.

        num_workers (``int``):

            The number of parallel workers.

    Returns:

        True, if all tasks succeeded.
    '''

    assert fragments.data.dtype == np.uint64

    shape = affs.shape[1:]
    context = daisy.Coordinate(context)

    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*affs.roi.dims(),
                         block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    return daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: agglomerate_in_block(
            affs,
            fragments,
            rag_provider,
            b,
            merge_function,
            threshold),
        lambda b: block_done(b, rag_provider),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')


def block_done(block, rag_provider):

    return (
        rag_provider.has_edges(block.write_roi) or
        rag_provider.num_nodes(block.write_roi) == 0)


def agglomerate_in_block(
        affs,
        fragments,
        rag_provider,
        block,
        merge_function,
        threshold):

    logger.info(
        "Agglomerating in block %s with context of %s",
        block.write_roi, block.read_roi)

    # get the sub-{affs, fragments, graph} to work on
    affs = affs.intersect(block.read_roi)
    fragments = fragments.to_ndarray(affs.roi, fill_value=0)
    rag = rag_provider[affs.roi]

    # waterz uses memory proportional to the max label in fragments, therefore
    # we relabel them here and use those
    fragments_relabelled, n, fragment_relabel_map = relabel(
        fragments,
        return_backwards_map=True)

    logger.debug("affs shape: %s", affs.shape)
    logger.debug("fragments shape: %s", fragments.shape)
    logger.debug("fragments num: %d", n)

    # convert affs to float32 ndarray with values between 0 and 1
    affs = affs.to_ndarray()[0:3]
    if affs.dtype == np.uint8:
        affs = affs.astype(np.float32)/255.0

    # So far, 'rag' does not contain any edges belonging to write_roi (there
    # might be a few edges from neighboring blocks, though). Run waterz until
    # threshold 0 to get the waterz RAG, which tells us which nodes are
    # neighboring. Use this to populate 'rag' with edges. Then run waterz for
    # the given threshold.

    # for efficiency, we create one waterz call with both thresholds
    generator = waterz.agglomerate(
        affs=affs,
        thresholds=[0, threshold],
        fragments=fragments_relabelled,
        scoring_function=merge_function,
        discretize_queue=256,
        return_merge_history=True,
        return_region_graph=True)

    # add edges to RAG
    _, _, initial_rag = next(generator)
    for edge in initial_rag:
        u, v = fragment_relabel_map[edge['u']], fragment_relabel_map[edge['v']]
        # this might overwrite already existing edges from neighboring blocks,
        # but that's fine, we only write attributes for edges within write_roi
        rag.add_edge(u, v, merge_score=None, agglomerated=True)

    # agglomerate fragments using affs
    _, merge_history, _ = next(generator)

    # cleanup generator
    for _, _, _ in generator:
        pass

    # create a merge tree from the merge history
    merge_tree = MergeTree(fragment_relabel_map)
    for merge in merge_history:

        a, b, c, score = merge['a'], merge['b'], merge['c'], merge['score']
        merge_tree.merge(
            fragment_relabel_map[a],
            fragment_relabel_map[b],
            fragment_relabel_map[c],
            score)

    # mark edges in original RAG with score at time of merging
    logger.debug("marking merged edges...")
    num_merged = 0
    for u, v, data in rag.edges(data=True):
        merge_score = merge_tree.find_merge(u, v)
        data['merge_score'] = merge_score
        if merge_score is not None:
            num_merged += 1

    logger.info("merged %d edges", num_merged)

    # write back results (only within write_roi)
    logger.debug("writing to DB...")
    rag.write_edges(block.write_roi)


def relabel(array, return_backwards_map=False, inplace=False):
    '''Relabel array, such that IDs are consecutive. Excludes 0.

    Args:

        array (ndarray):

                The array to relabel.

        return_backwards_map (``bool``, optional):

                If ``True``, return an ndarray that maps new labels (indices in
                the array) to old labels.

        inplace (``bool``, optional):

                Perform the replacement in-place on ``array``.

    Returns:

        A tuple ``(relabelled, n)``, where ``relabelled`` is the relabelled
        array and ``n`` the number of unique labels found.

        If ``return_backwards_map`` is ``True``, returns ``(relabelled, n,
        backwards_map)``.
    '''

    if array.size == 0:

        if return_backwards_map:
            return array, 0, []
        else:
            return array, 0

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:

        if return_backwards_map:
            return array, 0, [0]
        else:
            return array, 0

    n = len(old_labels)
    new_labels = np.arange(1, n + 1, dtype=array.dtype)

    replaced = replace_values(array, old_labels, new_labels, inplace=inplace)

    if return_backwards_map:

        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n


def replace_values(
        in_array, old_values, new_values, out_array=None, inplace=None):
    '''Replace each ``old_values`` in ``array`` with the corresponding
    ``new_values``. Other values are not changed.
    '''

    # `inplace` and `out_array` cannot be both specified
    if out_array is not None:
        assert (inplace is None) or (inplace is False)
    if inplace:
        out_array = in_array

    # `in_array` should always be a numpy array
    assert isinstance(in_array, (np.ndarray))

    # `old_values` and `new_values` are converted to numpy lists
    # if they are not already
    if not isinstance(old_values, (np.ndarray, np.generic)):
        old_values = np.array(old_values, dtype=in_array.dtype)
    if not isinstance(new_values, (np.ndarray, np.generic)):
        if out_array is None:
            # if `out_array` is None, its data type is guaranteed to be
            # same as `out_array`
            new_values = np.array(new_values, dtype=in_array.dtype)
        else:
            new_values = np.array(new_values, dtype=out_array.dtype)

    assert old_values.size == new_values.size
    assert in_array.dtype == old_values.dtype
    if out_array is not None:
        assert out_array.dtype == new_values.dtype
        assert out_array.size == in_array.size

    dtype = in_array.dtype

    min_value = in_array.min()
    max_value = in_array.max()
    value_range = max_value - min_value

    # can the relabeling be done with a values map?
    # this can only be done if `out_array` is not provided and when
    # `out_array` is provided and it _is_ `in_array`
    if (out_array is None or out_array is in_array) and value_range < 1024**3:

        valid_values = np.logical_and(
            old_values >= min_value,
            old_values <= max_value)
        old_values = old_values[valid_values]
        new_values = new_values[valid_values]

        # shift all values such that they start at 0
        offset = min_value
        in_array -= offset
        old_values -= offset

        # replace with a values map
        values_map = np.arange(
            start=min_value,
            stop=max_value + 1,
            dtype=dtype)
        values_map[old_values] = new_values

        inplace = out_array is in_array

        if inplace:

            in_array[:] = values_map[in_array]

        else:

            out_array = values_map[in_array]
            in_array += offset

        return out_array

    else:

        # replace using C++ implementation

        if out_array is None:
            out_array = in_array.copy()

        replace_values_inplace(
            np.ravel(in_array, order='A'),
            np.ravel(old_values, order='A'),
            np.ravel(new_values, order='A'),
            np.ravel(out_array, order='A'))

        return out_array
