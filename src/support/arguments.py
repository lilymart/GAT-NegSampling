import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    #DATA
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--interaction-mode", type=str, default="any", help="'any' for the union of all user-target metapaths or a 1-based metapath ID such as '1', '2', ...")
    parser.add_argument("--seed", type=int, default=42)

    #MODEL
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--threshold-metric", choices=["f1-binary", "f1-macro"], default="f1-macro")

    #LINK PREDICTION TASK
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--train-neg-ratio", type=float, default=1.0)
    parser.add_argument("--eval-neg-ratio", type=float, default=1.0)
    parser.add_argument("--negative-mode", choices=["none", "static", "dynamic"], default="static")
    parser.add_argument("--negative-strategy", choices=["random", "mixed", "mixed_self_adversarial", "mixed_structure_aware",], default="random")
    parser.add_argument("--disjoint-train-ratio", type=float, default=0.3)
    parser.add_argument("--path-mask", action="store_true", help="Remove the defining meta-path witness for positive user-target pairs.")

    return parser.parse_args()