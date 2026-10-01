from typing import Dict, List, Sequence, Tuple

import torch
from torch_geometric.data import HeteroData

from src.dataset.dataset_loader_relationwise import find_metapaths
from src.support.utils import get_target_type


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def get_interaction_metapath(
    data: HeteroData,
    dataset_name: str,
    interaction_mode: str,
    depth: int = 3,
) -> MetaPath:
    """
    Recupera esattamente il meta-path associato a interaction_mode.

    interaction_mode deve essere "1", "2", ..., non "any".
    """

    if not str(interaction_mode).isdigit():
        raise ValueError(
            "Path masking requires a single interaction mode: "
            "'1', '2', ..., not 'any'."
        )

    target_type = get_target_type(dataset_name)

    metapaths = find_metapaths(
        data=data,
        source_node_type="user",
        target_node_type=target_type,
        depth=depth,
        allow_repeated_node_types=False,
        excluded_node_types={"hashtag"},
    )

    # Stessa selezione usata nel loader.
    metapaths = [
        mp for mp in metapaths
        if len(mp) >= 2
        and all(
            edge_type[1] not in {
                "interacts_with",
                "rev_interacts_with",
            }
            for edge_type in mp
        )
    ]

    mp_id = int(interaction_mode)

    if mp_id < 1 or mp_id > len(metapaths):
        raise ValueError(
            f"Invalid interaction_mode={interaction_mode}. "
            f"Available IDs: 1..{len(metapaths)}."
        )

    return metapaths[mp_id - 1]


def get_positive_pairs(
    data: HeteroData,
    target_edge_type: EdgeType,
) -> torch.Tensor:
    """
    Estrae solo le coppie positive da edge_label_index.

    Va chiamata DOPO RandomLinkSplit.
    """

    store = data[target_edge_type]

    labels = (
        store.edge_label
        .detach()
        .view(-1)
        .cpu()
    )

    positive_mask = labels > 0.5

    return (
        store.edge_label_index[:, positive_mask]
        .detach()
        .cpu()
        .long()
        .contiguous()
    )


def merge_positive_pairs(
    *positive_edge_indices: torch.Tensor,
    num_target_nodes: int,
) -> torch.Tensor:
    """
    Union/deduplica più insiemi di coppie user-target.
    """

    edge_index = torch.cat(
        positive_edge_indices,
        dim=1,
    )

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


def _build_forward_csr(
    edge_index: torch.Tensor,
    num_source_nodes: int,
):
    """
    Indicizza una relazione per source node.

    Manteniamo anche gli edge ID originali perché dovremo
    sapere quali colonne di edge_index eliminare.
    """

    edge_index = (
        edge_index
        .detach()
        .cpu()
        .long()
    )

    src = edge_index[0]
    dst = edge_index[1]

    order = torch.argsort(
        src,
        stable=True,
    )

    src_sorted = src[order]
    dst_sorted = dst[order]

    degree = torch.bincount(
        src_sorted,
        minlength=num_source_nodes,
    )

    rowptr = torch.zeros(
        num_source_nodes + 1,
        dtype=torch.long,
    )

    rowptr[1:] = torch.cumsum(
        degree,
        dim=0,
    )

    return rowptr, dst_sorted, order


def _build_reverse_csr(
    edge_index: torch.Tensor,
    num_destination_nodes: int,
):
    """
    Indicizza una relazione per DESTINATION.

    Serve per percorrere il tail del meta-path al contrario:
        claim <- tweet <- reply
    """

    edge_index = (
        edge_index
        .detach()
        .cpu()
        .long()
    )

    src = edge_index[0]
    dst = edge_index[1]

    order = torch.argsort(
        dst,
        stable=True,
    )

    dst_sorted = dst[order]
    src_sorted = src[order]

    degree = torch.bincount(
        dst_sorted,
        minlength=num_destination_nodes,
    )

    rowptr = torch.zeros(
        num_destination_nodes + 1,
        dtype=torch.long,
    )

    rowptr[1:] = torch.cumsum(
        degree,
        dim=0,
    )

    return rowptr, src_sorted


def _csr_values(
    rowptr: torch.Tensor,
    values: torch.Tensor,
    node_id: int,
):
    start = int(rowptr[node_id])
    end = int(rowptr[node_id + 1])

    return values[start:end]


def _find_first_hop_candidates(
    target_node: int,
    reverse_tail_csrs,
):
    """
    Partendo dalla claim, percorre all'indietro tutto il meta-path
    TRANNE il primo hop.

    Esempio MP2:

    user -> reply -> tweet -> claim

    Partendo dalla claim otteniamo tutti i reply che possono
    completare:

        reply -> tweet -> claim

    A quel punto basta controllare quali di questi reply sono
    effettivamente collegati allo user.
    """

    current_nodes = torch.tensor(
        [target_node],
        dtype=torch.long,
    )

    for rowptr, predecessors in reversed(
        reverse_tail_csrs
    ):
        chunks = []

        for node in current_nodes.tolist():

            values = _csr_values(
                rowptr,
                predecessors,
                int(node),
            )

            if values.numel() > 0:
                chunks.append(values)

        if not chunks:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        current_nodes = torch.unique(
            torch.cat(chunks)
        )

    return current_nodes


def _find_first_hop_witness_edges(
    data: HeteroData,
    metapath: Sequence[EdgeType],
    positive_pairs: torch.Tensor,
):
    """
    Trova gli edge ID del primo hop che appartengono ad almeno
    un witness path completo per le coppie positive date.
    """

    first_edge_type = metapath[0]

    first_source_type = first_edge_type[0]

    first_rowptr, first_neighbors, first_edge_ids = (
        _build_forward_csr(
            edge_index=data[
                first_edge_type
            ].edge_index,
            num_source_nodes=data[
                first_source_type
            ].num_nodes,
        )
    )

    # Costruiamo gli indici inversi per r2, r3, ...
    reverse_tail_csrs = []

    for edge_type in metapath[1:]:

        destination_type = edge_type[2]

        reverse_tail_csrs.append(
            _build_reverse_csr(
                edge_index=data[
                    edge_type
                ].edge_index,
                num_destination_nodes=data[
                    destination_type
                ].num_nodes,
            )
        )

    # Molte coppie condividono la stessa claim.
    # Evitiamo di percorrere il tail migliaia di volte.
    target_cache: Dict[int, set] = {}

    edges_to_remove = set()
    pairs_with_witness = 0

    for user, target in positive_pairs.t().tolist():

        user = int(user)
        target = int(target)

        if target not in target_cache:

            candidates = (
                _find_first_hop_candidates(
                    target_node=target,
                    reverse_tail_csrs=(
                        reverse_tail_csrs
                    ),
                )
            )

            target_cache[target] = set(
                candidates.tolist()
            )

        valid_first_nodes = target_cache[target]

        start = int(first_rowptr[user])
        end = int(first_rowptr[user + 1])

        found = False

        for neighbor, edge_id in zip(
            first_neighbors[start:end].tolist(),
            first_edge_ids[start:end].tolist(),
        ):

            if neighbor in valid_first_nodes:

                edges_to_remove.add(
                    int(edge_id)
                )

                found = True

        if found:
            pairs_with_witness += 1

    return (
        torch.tensor(
            sorted(edges_to_remove),
            dtype=torch.long,
        ),
        pairs_with_witness,
    )


def _remove_edges(
    data: HeteroData,
    edge_type: EdgeType,
    edge_ids: torch.Tensor,
):
    """
    Remove selected edges and return the removed
    (source, destination) pairs.
    """

    edge_index = data[
        edge_type
    ].edge_index

    edge_ids_device = edge_ids.to(
        edge_index.device
    )

    removed_pairs = (
        edge_index[:, edge_ids_device]
        .detach()
        .cpu()
        .clone()
    )

    keep_mask = torch.ones(
        edge_index.size(1),
        dtype=torch.bool,
        device=edge_index.device,
    )

    keep_mask[edge_ids_device] = False

    data[
        edge_type
    ].edge_index = (
        edge_index[:, keep_mask]
        .contiguous()
    )

    return removed_pairs

def _edge_set(
    edge_index: torch.Tensor,
    num_destination_nodes: int,
):
    """
    Encode an edge relation as a set of integer IDs.
    Used only to identify the exact inverse relation.
    """

    edge_index = (
        edge_index
        .detach()
        .cpu()
        .long()
    )

    ids = (
        edge_index[0] * num_destination_nodes
        + edge_index[1]
    )

    return torch.unique(
        ids,
        sorted=True,
    )


def _find_exact_inverse_edge_type(
    data: HeteroData,
    edge_type: EdgeType,
):
    """
    Find the relation whose edge set is exactly the
    transpose of edge_type.

    Example:

        ('user', 'posted', 'tweet')

    matched against something like:

        ('tweet', ..., 'user')

    We do NOT rely on relation names.
    """

    src_type, _, dst_type = edge_type

    forward_edge_index = (
        data[edge_type]
        .edge_index
        .detach()
        .cpu()
        .long()
    )

    expected_reverse = (
        forward_edge_index
        .flip(0)
        .contiguous()
    )

    expected_set = _edge_set(
        expected_reverse,
        num_destination_nodes=(
            data[src_type].num_nodes
        ),
    )

    matches = []

    for candidate in data.edge_types:

        candidate_src, _, candidate_dst = (
            candidate
        )

        if (
            candidate_src != dst_type
            or candidate_dst != src_type
        ):
            continue

        candidate_set = _edge_set(
            data[candidate].edge_index,
            num_destination_nodes=(
                data[src_type].num_nodes
            ),
        )

        if (
            candidate_set.numel()
            == expected_set.numel()
            and torch.equal(
                candidate_set,
                expected_set,
            )
        ):
            matches.append(candidate)

    if len(matches) == 0:
        raise RuntimeError(
            f"No exact inverse relation found "
            f"for {edge_type}."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple exact inverse relations "
            f"found for {edge_type}: {matches}"
        )

    return matches[0]

def _remove_exact_pairs(
    data: HeteroData,
    edge_type: EdgeType,
    pairs_to_remove: torch.Tensor,
):
    """
    Remove exact (source, destination) pairs from an edge type.
    """

    edge_index = data[
        edge_type
    ].edge_index

    num_destination_nodes = int(
        data[edge_type[2]].num_nodes
    )

    current_ids = (
        edge_index[0].detach().cpu().long()
        * num_destination_nodes
        + edge_index[1].detach().cpu().long()
    )

    remove_ids = (
        pairs_to_remove[0].long()
        * num_destination_nodes
        + pairs_to_remove[1].long()
    )

    remove_ids = torch.unique(
        remove_ids
    )

    remove_mask_cpu = torch.isin(
        current_ids,
        remove_ids,
    )

    keep_mask = (
        ~remove_mask_cpu
    ).to(edge_index.device)

    removed_count = int(
        remove_mask_cpu.sum().item()
    )

    data[
        edge_type
    ].edge_index = (
        edge_index[:, keep_mask]
        .contiguous()
    )

    return removed_count


def mask_witness_paths(
    data: HeteroData,
    metapath: Sequence[EdgeType],
    positive_pairs: torch.Tensor,
    split_name: str,
):
    """
    Rompe tutti i witness path delle coppie positive fornite
    eliminando il PRIMO hop.

    Non modifica edge_label_index né la relazione target.
    """

    masked_data = data.clone()

    first_edge_type = metapath[0]

    edges_before = masked_data[
        first_edge_type
    ].edge_index.size(1)

    edge_ids, witnesses_before = (
        _find_first_hop_witness_edges(
            data=masked_data,
            metapath=metapath,
            positive_pairs=positive_pairs,
        )
    )

    reverse_edge_type = (
        _find_exact_inverse_edge_type(
            data=masked_data,
            edge_type=first_edge_type,
        )
    )

    removed_forward_pairs = _remove_edges(
        data=masked_data,
        edge_type=first_edge_type,
        edge_ids=edge_ids,
    )

    removed_reverse_pairs = (
        removed_forward_pairs
        .flip(0)
        .contiguous()
    )

    reverse_edges_removed = (
        _remove_exact_pairs(
            data=masked_data,
            edge_type=reverse_edge_type,
            pairs_to_remove=removed_reverse_pairs,
        )
    )

    # Sanity check fondamentale:
    # dopo il masking non deve sopravvivere nessun witness.
    _, witnesses_after = (
        _find_first_hop_witness_edges(
            data=masked_data,
            metapath=metapath,
            positive_pairs=positive_pairs,
        )
    )

    diagnostics = {
        "split": split_name,
        "positive_pairs": int(
            positive_pairs.size(1)
        ),
        "pairs_with_witness_before": int(
            witnesses_before
        ),
        "edges_removed": int(
            edge_ids.numel()
        ),
        "edges_before": int(
            edges_before
        ),
        "edges_after": int(
            masked_data[
                first_edge_type
            ].edge_index.size(1)
        ),
        "fraction_edges_removed": (
            edge_ids.numel() / edges_before
            if edges_before > 0
            else 0.0
        ),
        "forward_edge_type":
            first_edge_type,

        "reverse_edge_type":
            reverse_edge_type,

        "forward_edges_removed":
            int(edge_ids.numel()),

        "reverse_edges_removed":
            int(reverse_edges_removed),
        "pairs_with_witness_after": int(
            witnesses_after
        ),
    }

    if witnesses_after != 0:
        raise RuntimeError(
            f"Path masking failed on {split_name}: "
            f"{witnesses_after} positive pairs "
            "still have a complete witness path."
        )

    return masked_data, diagnostics