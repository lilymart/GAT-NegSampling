import gc
import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from src.support.utils import get_base_dir, get_target_type

from collections import defaultdict
from typing import Dict, List, Set, Tuple


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def _load_tensor(path: str) -> torch.Tensor:
    """Load a tensor on CPU, using memory mapping when supported."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (TypeError, RuntimeError):
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )


def build_heterodata(dataset_name):
    data = HeteroData()

    heterodata_dir = os.path.join(get_base_dir(), dataset_name, "heterodata")
    feats_dir = os.path.join(heterodata_dir, "features")
    edges_dir = os.path.join(heterodata_dir, "edgelists")

    # Load node features
    files_feats = [
        f for f in os.listdir(feats_dir)
        if os.path.isfile(os.path.join(feats_dir, f))
    ]

    for fname in files_feats:
        n_type = fname[:-3]  # remove ".pt"

        features = _load_tensor(os.path.join(feats_dir, fname))

        if features.dtype == torch.float64:
            features = features.float()

        data[n_type].x = features

    # Optional labels, not required for link prediction
    target_type = get_target_type(dataset_name)
    label_path = os.path.join(heterodata_dir, f"{target_type}_labels.pt")
    if os.path.exists(label_path):
        data[target_type].y = _load_tensor(label_path)

    # Load original heterogeneous edges
    files_edges = [
        f for f in os.listdir(edges_dir)
        if os.path.isfile(os.path.join(edges_dir, f))
    ]

    for fname in files_edges:
        n_type_src, n_type_tgt, e_type = _extract_edge_info(fname)
        edge_index = _load_tensor(os.path.join(edges_dir, fname)).long()

        data[n_type_src, e_type, n_type_tgt].edge_index = edge_index

    return data

def _extract_edge_info(fname):
    last_dot = fname.rfind(".")
    fname_base = fname[:last_dot]

    first_underscore = fname_base.find("_")
    last_underscore = fname_base.rfind("_")

    n_type_src = fname_base[:first_underscore]
    e_type = fname_base[first_underscore + 1:last_underscore]
    n_type_tgt = fname_base[last_underscore + 1:]

    return n_type_src, n_type_tgt, e_type



""" Return all metapaths from source_node_type to target_node_type with length from 1 to depth."""
def find_metapaths(
    data: HeteroData,
    source_node_type: str,
    target_node_type: str,
    depth: int, #max number of edges in the metapath
    allow_repeated_node_types: bool = True, #if False, a node type can appear only once in the metapath
    excluded_node_types: Set[str] = None #eventuali tipi di nodo da escludere dai metapaths
) -> List[MetaPath]:

    if depth < 1:
        raise ValueError("'depth' deve essere maggiore o uguale a 1.")

    node_types, edge_types = data.metadata()

    if source_node_type not in node_types:
        raise ValueError(
            f"Tipo sorgente '{source_node_type}' non presente. "
            f"Tipi disponibili: {node_types}"
        )

    if target_node_type not in node_types:
        raise ValueError(
            f"Tipo target '{target_node_type}' non presente. "
            f"Tipi disponibili: {node_types}"
        )

    if excluded_node_types is None:
        excluded_node_types = set()
    else:
        excluded_node_types = set(excluded_node_types)

    unknown_excluded_node_types = (
            excluded_node_types - set(node_types)
    )

    if unknown_excluded_node_types:
        raise ValueError(
            "Tipi di nodo da escludere non presenti nel grafo: "
            f"{sorted(unknown_excluded_node_types)}"
        )

    if source_node_type in excluded_node_types:
        raise ValueError(
            f"Il tipo sorgente '{source_node_type}' non può essere escluso."
        )

    if target_node_type in excluded_node_types:
        raise ValueError(
            f"Il tipo target '{target_node_type}' non può essere escluso."
        )

    adjacency: Dict[str, List[EdgeType]] = defaultdict(list)

    for edge_type in edge_types:
        source_type, _, destination_type = edge_type

        if (source_type in excluded_node_types or destination_type in excluded_node_types):
            continue

        adjacency[source_type].append(edge_type)

    # Rende deterministico l'ordine dei risultati.
    for source_type in adjacency:
        adjacency[source_type].sort()

    metapaths: List[MetaPath] = []

    _find_metapaths_dfs(
        current_node_type=source_node_type,
        target_node_type=target_node_type,
        depth=depth,
        adjacency=adjacency,
        current_metapath=[],
        visited_node_types={source_node_type},
        metapaths=metapaths,
        allow_repeated_node_types=allow_repeated_node_types,
    )

    return metapaths


""" Funzione ricorsiva di supporto per l'esplorazione dei metapath. """
def _find_metapaths_dfs(
    current_node_type: str,
    target_node_type: str,
    depth: int,
    adjacency: Dict[str, List[EdgeType]],
    current_metapath: MetaPath,
    visited_node_types: Set[str],
    metapaths: List[MetaPath],
    allow_repeated_node_types: bool,
) -> None:

    if len(current_metapath) >= depth:
        return

    for edge_type in adjacency.get(current_node_type, []):
        _, _, destination_node_type = edge_type

        if (
            not allow_repeated_node_types
            and destination_node_type in visited_node_types
        ):
            continue

        new_metapath = current_metapath + [edge_type]

        if destination_node_type == target_node_type:
            metapaths.append(new_metapath)

        if allow_repeated_node_types:
            new_visited_node_types = visited_node_types
        else:
            new_visited_node_types = (
                visited_node_types | {destination_node_type}
            )

        _find_metapaths_dfs(
            current_node_type=destination_node_type,
            target_node_type=target_node_type,
            depth=depth,
            adjacency=adjacency,
            current_metapath=new_metapath,
            visited_node_types=new_visited_node_types,
            metapaths=metapaths,
            allow_repeated_node_types=allow_repeated_node_types,
        )


def _atomic_torch_save(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor atomically to avoid leaving a partial cache file."""
    temporary_path = f"{path}.tmp"
    torch.save(tensor, temporary_path)
    os.replace(temporary_path, path)


def _deduplicate_edge_index(
    edge_index: torch.Tensor,
    num_target_nodes: int,
) -> torch.Tensor:
    """Remove duplicate source-target pairs using a compact 1-D encoding."""
    edge_index = edge_index.detach().cpu().long().contiguous()

    pair_ids = (
        edge_index[0] * num_target_nodes
        + edge_index[1]
    )
    pair_ids = torch.unique(pair_ids)

    return torch.stack(
        [
            pair_ids // num_target_nodes,
            pair_ids % num_target_nodes,
        ],
        dim=0,
    ).contiguous()


def _compose_metapath_exact(
    data: HeteroData,
    metapath: MetaPath,
    batch_size: int = 50000,
) -> torch.Tensor:
    """
    Compose a metapath exactly using edge-list joins.

    Unlike AddMetaPaths, this function does not perform sparse matrix
    multiplications. Intermediate user-node pairs are processed in batches,
    deduplicated after every relation, and kept on CPU.

    The returned edge_index links the source type of the first relation to
    the target type of the last relation.
    """
    if len(metapath) < 1:
        raise ValueError("Metapath cannot be empty.")

    first_edge_type = metapath[0]
    current_node_type = first_edge_type[2]

    first_edge_index = _deduplicate_edge_index(
        edge_index=data[first_edge_type].edge_index,
        num_target_nodes=data[current_node_type].num_nodes,
    )

    original_source_nodes = first_edge_index[0]
    current_nodes = first_edge_index[1]

    del first_edge_index

    for step, edge_type in enumerate(
        metapath[1:],
        start=2,
    ):
        relation_source_type, _, relation_target_type = edge_type

        if relation_source_type != current_node_type:
            raise ValueError(
                f"Invalid metapath continuity: expected source type "
                f"'{current_node_type}', found '{relation_source_type}' "
                f"in relation {edge_type}."
            )

        num_relation_source_nodes = data[
            relation_source_type
        ].num_nodes
        num_relation_target_nodes = data[
            relation_target_type
        ].num_nodes

        relation_edge_index = _deduplicate_edge_index(
            edge_index=data[edge_type].edge_index,
            num_target_nodes=num_relation_target_nodes,
        )

        relation_sources = relation_edge_index[0]
        relation_targets = relation_edge_index[1]

        sorting_indices = torch.argsort(
            relation_sources,
            stable=True,
        )
        relation_sources = relation_sources[sorting_indices]
        relation_targets = relation_targets[sorting_indices]

        source_degrees = torch.bincount(
            relation_sources,
            minlength=num_relation_source_nodes,
        )

        row_pointer = torch.zeros(
            num_relation_source_nodes + 1,
            dtype=torch.long,
        )
        row_pointer[1:] = torch.cumsum(
            source_degrees,
            dim=0,
        )

        next_pair_ids = torch.empty(
            0,
            dtype=torch.long,
        )

        num_current_pairs = current_nodes.numel()

        for batch_start in range(
            0,
            num_current_pairs,
            batch_size,
        ):
            batch_end = min(
                batch_start + batch_size,
                num_current_pairs,
            )

            batch_original_sources = original_source_nodes[
                batch_start:batch_end
            ]
            batch_current_nodes = current_nodes[
                batch_start:batch_end
            ]

            batch_degrees = source_degrees[
                batch_current_nodes
            ]
            valid_mask = batch_degrees > 0

            if not valid_mask.any():
                continue

            batch_original_sources = batch_original_sources[
                valid_mask
            ]
            batch_current_nodes = batch_current_nodes[
                valid_mask
            ]
            batch_degrees = batch_degrees[
                valid_mask
            ]

            total_output_pairs = int(
                batch_degrees.sum().item()
            )

            relation_starts = row_pointer[
                batch_current_nodes
            ]

            repeated_relation_starts = torch.repeat_interleave(
                relation_starts,
                batch_degrees,
            )

            output_group_starts = (
                torch.cumsum(batch_degrees, dim=0)
                - batch_degrees
            )

            repeated_output_group_starts = torch.repeat_interleave(
                output_group_starts,
                batch_degrees,
            )

            offsets = (
                torch.arange(
                    total_output_pairs,
                    dtype=torch.long,
                )
                - repeated_output_group_starts
            )

            target_positions = (
                repeated_relation_starts
                + offsets
            )

            joined_target_nodes = relation_targets[
                target_positions
            ]

            joined_original_sources = torch.repeat_interleave(
                batch_original_sources,
                batch_degrees,
            )

            batch_pair_ids = (
                joined_original_sources
                * num_relation_target_nodes
                + joined_target_nodes
            )
            batch_pair_ids = torch.unique(
                batch_pair_ids
            )

            next_pair_ids = torch.unique(
                torch.cat(
                    [
                        next_pair_ids,
                        batch_pair_ids,
                    ],
                    dim=0,
                )
            )

            del (
                batch_original_sources,
                batch_current_nodes,
                batch_degrees,
                relation_starts,
                repeated_relation_starts,
                output_group_starts,
                repeated_output_group_starts,
                offsets,
                target_positions,
                joined_target_nodes,
                joined_original_sources,
                batch_pair_ids,
            )

        if next_pair_ids.numel() == 0:
            return torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        original_source_nodes = (
            next_pair_ids
            // num_relation_target_nodes
        )
        current_nodes = (
            next_pair_ids
            % num_relation_target_nodes
        )
        current_node_type = relation_target_type

        del (
            relation_edge_index,
            relation_sources,
            relation_targets,
            sorting_indices,
            source_degrees,
            row_pointer,
            next_pair_ids,
        )

        gc.collect()

        print(
            f"  Composed relation {step}/{len(metapath)} "
            f"({edge_type}) - "
            f"unique source-current pairs: "
            f"{current_nodes.numel()}"
        )

    return torch.stack(
        [
            original_source_nodes,
            current_nodes,
        ],
        dim=0,
    ).contiguous()


def _get_single_metapath_cache_path(
    cache_dir: str,
    metapath_index: int,
) -> str:
    return os.path.join(
        cache_dir,
        f"metapath_{metapath_index:02d}_exact.pt",
    )


def _merge_metapaths_into_target_relation(
    data: HeteroData,
    metapaths: List[MetaPath],
    target_edge_type: EdgeType,
    reverse_target_edge_type: EdgeType,
    cache_dir: str,
    batch_size: int = 50000,
) -> HeteroData:
    """
    Build the target relation as the exact union of all metapath relations.

    Each metapath is composed through batched edge-list joins and cached
    separately. This avoids native sparse matrix multiplication and allows
    interrupted preprocessing to resume from completed metapaths.
    """
    if not metapaths:
        raise ValueError("Metapath list is empty.")

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    target_node_type = target_edge_type[2]
    num_target_nodes = data[
        target_node_type
    ].num_nodes

    target_pair_ids = torch.empty(
        0,
        dtype=torch.long,
    )

    for index, metapath in enumerate(
        metapaths,
        start=1,
    ):
        """print(
            f"\nProcessing metapath "
            f"{index}/{len(metapaths)}:"
        )
        for edge_type in metapath:
            print(f"  {edge_type}")"""

        metapath_cache_path = (
            _get_single_metapath_cache_path(
                cache_dir=cache_dir,
                metapath_index=index,
            )
        )

        if os.path.exists(
            metapath_cache_path
        ):
            """print(
                "  Loading cached metapath relation: "
                f"{metapath_cache_path}"
            )"""
            current_edge_index = _load_tensor(
                metapath_cache_path
            ).long()
        else:
            current_edge_index = (
                _compose_metapath_exact(
                    data=data,
                    metapath=metapath,
                    batch_size=batch_size,
                )
            )

            _atomic_torch_save(
                current_edge_index,
                metapath_cache_path,
            )

            """print(
                "  Metapath relation cached in: "
                f"{metapath_cache_path}"
            )"""

        if current_edge_index.numel() > 0:
            current_pair_ids = (
                current_edge_index[0]
                * num_target_nodes
                + current_edge_index[1]
            )

            target_pair_ids = torch.unique(
                torch.cat(
                    [
                        target_pair_ids,
                        current_pair_ids,
                    ],
                    dim=0,
                )
            )

            del current_pair_ids

        del current_edge_index
        gc.collect()

        """print(
            f"Processed metapath "
            f"{index}/{len(metapaths)} - "
            f"unique target links: "
            f"{target_pair_ids.numel()}"
        )"""

    if target_pair_ids.numel() == 0:
        raise ValueError(
            "No target links were generated "
            "from the provided metapaths."
        )

    target_edge_index = torch.stack(
        [
            target_pair_ids
            // num_target_nodes,
            target_pair_ids
            % num_target_nodes,
        ],
        dim=0,
    ).contiguous()

    data[
        target_edge_type
    ].edge_index = target_edge_index

    data[
        reverse_target_edge_type
    ].edge_index = target_edge_index.flip(
        0
    ).contiguous()

    return data

def _get_target_relation_cache_path(
    dataset_name: str,
    target_node_type: str,
    depth: int,
) -> str:
    cache_dir = os.path.join(
        get_base_dir(),
        dataset_name,
        "heterodata",
        "processed",
    )

    os.makedirs(cache_dir, exist_ok=True)

    return os.path.join(
        cache_dir,
        (
            f"user_interacts_with_{target_node_type}_"
            f"depth{depth}_no_hashtag_exact_v2.pt"
        ),
    )


def build_heterodata_full(
    dataset_name,
    interaction_mode="any",
    force_rebuild_target_relation=False,
    metapath_batch_size=50000,
):
    data = build_heterodata(dataset_name=dataset_name)

    target_node_type = get_target_type(dataset_name)
    target_edge_type = (
        "user",
        "interacts_with",
        target_node_type,
    )
    reverse_target_edge_type = (
        target_node_type,
        "rev_interacts_with",
        "user",
    )

    if interaction_mode != "any":
        return (
            data,
            target_edge_type,
            reverse_target_edge_type,
        )

    metapath_depth = 3

    cache_path = _get_target_relation_cache_path(
        dataset_name=dataset_name,
        target_node_type=target_node_type,
        depth=metapath_depth,
    )

    if (
        os.path.exists(cache_path)
        and not force_rebuild_target_relation
    ):
        #print(f"Loading cached target relation: {cache_path}")

        target_edge_index = _load_tensor(cache_path).long()

        data[target_edge_type].edge_index = target_edge_index
        data[reverse_target_edge_type].edge_index = (
            target_edge_index.flip(0).contiguous()
        )

        return (
            data,
            target_edge_type,
            reverse_target_edge_type,
        )

    metapaths_from_user = find_metapaths(
        data=data,
        source_node_type="user",
        target_node_type=target_node_type,
        depth=metapath_depth,
        allow_repeated_node_types=False,
        excluded_node_types={"hashtag"},
    )

    metapaths_from_user = [
        metapath
        for metapath in metapaths_from_user
        if len(metapath) >= 2
    ]

    """print(
        f"Building target relation from "
        f"{len(metapaths_from_user)} metapaths."
    )"""

    metapath_cache_dir = os.path.join(
        get_base_dir(),
        dataset_name,
        "heterodata",
        "processed",
        "metapaths_exact_v2",
    )

    data = _merge_metapaths_into_target_relation(
        data=data,
        metapaths=metapaths_from_user,
        target_edge_type=target_edge_type,
        reverse_target_edge_type=reverse_target_edge_type,
        cache_dir=metapath_cache_dir,
        batch_size=metapath_batch_size,
    )

    target_edge_index = (
        data[target_edge_type]
        .edge_index
        .detach()
        .cpu()
        .contiguous()
    )

    _atomic_torch_save(target_edge_index, cache_path)

    #print(f"Target relation cached in: {cache_path}")

    return (
        data,
        target_edge_type,
        reverse_target_edge_type,
    )


def split_heterodata(
        data: HeteroData,
        target_edge_type: EdgeType,
        reverse_target_edge_type: EdgeType,
        val_ratio: float = 0.15,
        test_ratio: float = 0.25,
        eval_neg_sampling_ratio: float = 1.0,
        disjoint_train_ratio: float = 0.3
) -> Tuple[HeteroData, HeteroData, HeteroData]:

    if target_edge_type not in data.edge_types:
        raise ValueError(
            f"La relazione target {target_edge_type} non è presente. "
            f"Relazioni disponibili: {data.edge_types}"
        )

    if reverse_target_edge_type not in data.edge_types:
        raise ValueError(
            f"La relazione inversa {reverse_target_edge_type} non è presente. "
            f"Relazioni disponibili: {data.edge_types}"
        )

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        disjoint_train_ratio=disjoint_train_ratio, #0.0, 0.2, 0.3... #0.3 top mumin e 0.2 top politifact con negativi statici (add_negative_train_samples=True)
        neg_sampling_ratio=eval_neg_sampling_ratio,
        add_negative_train_samples=False, #False non mette negativi, serve per negativi dinamici. true li mette statici
        edge_types=target_edge_type,
        rev_edge_types=reverse_target_edge_type
    )

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data



if __name__ == '__main__':
    dataset_name = "politifact"
    data, rel, rev_rel = build_heterodata_full(dataset_name=dataset_name)
    print(data)

    train_data, val_data, test_data = split_heterodata(data, rel, rev_rel)


    """
    dataset_name = "politifact"
    data = build_heterodata(dataset_name)
    source_node_type = "user"
    target_node_type = get_target_type(dataset_name)
    depth = 3
    excluded_node_types={"hashtag"}

    mps = find_metapaths(data=data,
                         source_node_type=source_node_type,
                         target_node_type=target_node_type,
                         depth=depth,
                         allow_repeated_node_types=False,
                         excluded_node_types=excluded_node_types)

    for mp in mps:
        print(mp)
    """
