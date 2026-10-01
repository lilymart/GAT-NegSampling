import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths, RandomLinkSplit

from src.support.utils import get_base_dir, get_target_type

from collections import defaultdict
from typing import Dict, List, Set, Tuple


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


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

        data[n_type].x = torch.load(os.path.join(feats_dir, fname))

    # Optional labels, not required for link prediction
    target_type = get_target_type(dataset_name)
    label_path = os.path.join(heterodata_dir, f"{target_type}_labels.pt")
    if os.path.exists(label_path):
        data[target_type].y = torch.load(label_path)

    # Load original heterogeneous edges
    files_edges = [
        f for f in os.listdir(edges_dir)
        if os.path.isfile(os.path.join(edges_dir, f))
    ]

    for fname in files_edges:
        n_type_src, n_type_tgt, e_type = _extract_edge_info(fname)
        edge_index = torch.load(os.path.join(edges_dir, fname))

        if edge_index.dtype == torch.float64:
            edge_index = edge_index.to(torch.int64)

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


"""Costruisce un'unica relazione target come unione degli archi generati da tutti i metapath 
    del tipo: user interacts_with news/claim.
    Un arco user-news è inserito se la coppia è collegata da almeno uno dei metapath forniti."""
def _merge_metapaths_into_target_relation(
    data: HeteroData,
    metapaths: List[MetaPath],
    target_edge_type: EdgeType,
    reverse_target_edge_type: EdgeType,
) -> HeteroData:

    if not metapaths:
        raise ValueError("Metapath list is empty")

    original_edge_types = set(data.edge_types)

    # AddMetaPaths genera una relazione metapath_i per ogni metapath.
    data = AddMetaPaths(metapaths)(data)

    generated_metapath_edge_types = [
        edge_type
        for edge_type in data.edge_types
        if edge_type not in original_edge_types
    ]

    if not generated_metapath_edge_types:
        raise ValueError(
            "AddMetaPaths non ha generato alcun nuovo edge type."
        )

    metapath_edge_indices = []

    for edge_type in generated_metapath_edge_types:
        source_type, _, destination_type = edge_type

        if (
            source_type == target_edge_type[0]
            and destination_type == target_edge_type[2]
        ):
            metapath_edge_indices.append(
                data[edge_type].edge_index
            )

    if not metapath_edge_indices:
        raise ValueError(
            "Nessuna relazione generata dai metapath collega "
            f"{target_edge_type[0]} a {target_edge_type[2]}."
        )

    # Unisce gli archi prodotti dai diversi metapath.
    target_edge_index = torch.cat(
        metapath_edge_indices,
        dim=1,
    )

    # Elimina le coppie user-news duplicate.
    target_edge_index = torch.unique(
        target_edge_index,
        dim=1,
    )

    data[target_edge_type].edge_index = target_edge_index

    # Costruisce l'esatta relazione inversa.
    data[reverse_target_edge_type].edge_index = (
        target_edge_index.flip(0)
    )

    # Le relazioni metapath_i erano temporanee: ora possono essere eliminate.
    for edge_type in generated_metapath_edge_types:
        del data[edge_type]

    return data



def build_heterodata_full(dataset_name, interaction_mode="binary"):
    data = build_heterodata(dataset_name=dataset_name)

    target_node_type = get_target_type(dataset_name)

    target_edge_type = ("user", "interacts_with", target_node_type)

    reverse_target_edge_type = (target_node_type, "rev_interacts_with", "user")

    metapaths_from_user = find_metapaths(
        data=data,
        source_node_type="user",
        target_node_type=target_node_type,
        depth=3,
        allow_repeated_node_types=False,
        excluded_node_types = {"hashtag"}
    )

    # AddMetaPaths richiede metapath composti da almeno due archi.
    metapaths_from_user = [
        metapath
        for metapath in metapaths_from_user
        if len(metapath) >= 2
    ]

    if interaction_mode=="binary":
        data = _merge_metapaths_into_target_relation(
            data=data,
            metapaths=metapaths_from_user,
            target_edge_type=target_edge_type,
            reverse_target_edge_type=reverse_target_edge_type,
        )

    return data, target_edge_type, reverse_target_edge_type

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

    #train_data, val_data, test_data = split_heterodata(data, rel, rev_rel)


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


