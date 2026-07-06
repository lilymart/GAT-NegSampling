import os
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.transforms import AddMetaPaths, RandomLinkSplit

from src.support.utils import get_base_dir, load_embeddings

"""
Select the edge type to predict.

For binary link prediction:
- MuMiN: user -> claim interaction
- Politifact: user -> news interaction
"""
def get_target_link_type(dataset_name, interaction_mode="binary"):

    dataset_name = dataset_name.lower()

    if dataset_name == "mumin":
        if interaction_mode == "binary":
            return ("user", "interacts_with", "claim")
        elif interaction_mode == "posted":
            return ("user", "posted_claim_tweet", "claim")
        elif interaction_mode == "replied":
            return ("user", "replied_to_claim", "claim")
        elif interaction_mode == "quoted":
            return ("user", "quoted_claim", "claim")

    if dataset_name == "politifact":
        if interaction_mode == "binary":
            return ("user", "interacts_with", "news")
        elif interaction_mode == "posted":
            return ("user", "posted_news_tweet", "news")
        elif interaction_mode == "mentioned":
            return ("user", "mentioned_news", "news")

    raise ValueError(
        f"No target link type for dataset={dataset_name}, "
        f"interaction_mode={interaction_mode}"
    )


def build_heterodata_for_link_prediction(
    dataset_name,
    target_type,
    interaction_mode="binary",
    add_metapaths=True
):
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

    # Build explicit user -> claim/news interaction edges
    data = add_interaction_edges(
        data=data,
        dataset_name=dataset_name,
        target_type=target_type,
        interaction_mode=interaction_mode
    )

    if add_metapaths:
        metapaths = _get_metapaths(dataset_name)
        data = AddMetaPaths(metapaths, weighted=True)(data)

    data.embeddings = load_embeddings(dataset_name, target_type)

    return data


def build_link_prediction_splits(
    dataset_name,
    target_type,
    interaction_mode="binary",
    val_ratio=0.15,
    test_ratio=0.25,
    neg_sampling_ratio=1.0
):
    data = build_heterodata_for_link_prediction(
        dataset_name=dataset_name,
        target_type=target_type,
        interaction_mode=interaction_mode
    )

    edge_type = get_target_link_type(dataset_name, interaction_mode)

    reverse_edge_type = _infer_reverse_edge_type(data, edge_type)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=True,
        edge_types=[edge_type],
        rev_edge_types=[reverse_edge_type] if reverse_edge_type else None
    )

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data, edge_type


"""
Creates direct user -> target links for link prediction.

This assumes the graph contains:
user -> tweet edges and tweet -> claim/news edges.
"""
def add_interaction_edges(data, dataset_name, target_type, interaction_mode="binary"):

    dataset_name = dataset_name.lower()

    if dataset_name == "mumin":
        target_edge_type = ("user", "interacts_with", "claim")

        # Basic interaction: user posted a tweet that discusses a claim.
        posted_edge = ("user", "posted", "tweet")
        discusses_edge = ("tweet", "discusses", "claim")

    elif dataset_name == "politifact":
        target_edge_type = ("user", "interacts_with", "news")

        posted_edge = ("user", "posted", "tweet")
        discusses_edge = ("tweet", "discusses", "news")

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    if interaction_mode != "binary":
        # For relation-specific prediction, create separate relation names.
        # You can extend this block for comment/share/quote/reply relations.
        target_edge_type = get_target_link_type(dataset_name, interaction_mode)

    if posted_edge not in data.edge_types:
        raise ValueError(f"Missing edge type: {posted_edge}")

    if discusses_edge not in data.edge_types:
        raise ValueError(f"Missing edge type: {discusses_edge}")

    user_tweet = data[posted_edge].edge_index
    tweet_target = data[discusses_edge].edge_index

    interaction_edge_index = compose_user_target_edges(
        user_tweet_edge_index=user_tweet,
        tweet_target_edge_index=tweet_target
    )

    data[target_edge_type].edge_index = interaction_edge_index

    # Add reverse relation for message passing and proper link split.
    rev_edge_type = (target_type, "rev_interacts_with", "user")
    data[rev_edge_type].edge_index = interaction_edge_index.flip(0)

    return data


"""
Compose:
    user -> tweet
    tweet -> target
into:
    user -> target
"""
def compose_user_target_edges(user_tweet_edge_index, tweet_target_edge_index):

    user_to_tweets = user_tweet_edge_index.t().tolist()
    tweet_to_targets = tweet_target_edge_index.t().tolist()

    tweet_to_target_dict = {}
    for tweet, target in tweet_to_targets:
        tweet_to_target_dict.setdefault(tweet, []).append(target)

    edges = []
    for user, tweet in user_to_tweets:
        if tweet in tweet_to_target_dict:
            for target in tweet_to_target_dict[tweet]:
                edges.append((user, target))

    if len(edges) == 0:
        raise ValueError("No user-target interaction edges were created.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Remove duplicate interactions
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def _infer_reverse_edge_type(data, edge_type):
    src, rel, dst = edge_type

    candidates = [
        (dst, f"rev_{rel}", src),
        (dst, "rev_interacts_with", src),
        (dst, f"is_{rel}_by", src),
    ]

    for candidate in candidates:
        if candidate in data.edge_types:
            return candidate

    return None


def _extract_edge_info(fname):
    last_dot = fname.rfind(".")
    fname_base = fname[:last_dot]

    first_underscore = fname_base.find("_")
    last_underscore = fname_base.rfind("_")

    n_type_src = fname_base[:first_underscore]
    e_type = fname_base[first_underscore + 1:last_underscore]
    n_type_tgt = fname_base[last_underscore + 1:]

    return n_type_src, n_type_tgt, e_type


def _get_metapaths(dataset_name):
    if dataset_name.lower() == "mumin":
        return [
            [
                ("claim", "is_discussed_by", "tweet"),
                ("tweet", "is_posted_by", "user"),
                ("user", "posted", "tweet"),
                ("tweet", "discusses", "claim")
            ],
            [
                ("claim", "is_discussed_by", "tweet"),
                ("tweet", "has_hashtag", "hashtag"),
                ("hashtag", "is_hashtag_of", "tweet"),
                ("tweet", "discusses", "claim")
            ],
            [
                ("claim", "is_discussed_by", "tweet"),
                ("tweet", "is_replied_by", "reply"),
                ("reply", "reply_to", "tweet"),
                ("tweet", "discusses", "claim")
            ],
            [
                ("claim", "is_discussed_by", "tweet"),
                ("tweet", "is_quoted_by", "reply"),
                ("reply", "quote_of", "tweet"),
                ("tweet", "discusses", "claim")
            ]
        ]

    return [
        [
            ("news", "is_discussed_by", "tweet"),
            ("tweet", "is_posted_by", "user"),
            ("user", "posted", "tweet"),
            ("tweet", "discusses", "news")
        ],
        [
            ("news", "is_discussed_by", "tweet"),
            ("tweet", "has_hashtag", "hashtag"),
            ("hashtag", "is_hashtag_of", "tweet"),
            ("tweet", "discusses", "news")
        ],
        [
            ("news", "is_discussed_by", "tweet"),
            ("tweet", "is_posted_by", "user"),
            ("user", "mentions", "user"),
            ("user", "posted", "tweet"),
            ("tweet", "discusses", "news")
        ]
    ]