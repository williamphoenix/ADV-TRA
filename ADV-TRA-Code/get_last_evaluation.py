#!/usr/bin/env python3
# eval_per_task.py  ––  report per‑task accuracies for each ctx‑checkpoint
import os, torch, argparse, numpy as np

from utils.params import options
from utils.params.param_values import set_method_options, set_default_values
from utils.many_loads import get_context_set
from attacks import define_models as define
from attacks.eval import evaluate
from utils import utils                      # load_checkpoint()

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120, edgeitems=3)


# ---------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------
def parse_cli():
    parser = options.define_args(filename="get_last_evaluation",
                                 description="Accuracy of ctx‑checkpoints")
    parser = options.add_general_options(parser)
    parser = options.add_model_options(parser)
    parser = options.add_problem_options(parser)
    parser = options.add_train_options(parser)
    parser.add_argument('--ckpt-dir', required=True,
                        help='Directory that holds *-ctx*.pt checkpoints')
    parser.set_defaults(
        experiment='CIFAR10',
        c_type='normal',
        scenario='task',
        depth=2,
        conv_type='resNet',
        fc_units=128,
        fc_layers=3,
        seed=43,
        save=False,
        visdom=False,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# 2. main
# ---------------------------------------------------------------------
def main():
    args = parse_cli()
    set_method_options(args)
    set_default_values(args, also_hyper_params=True)

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # datasets -----------------------------------------------------------------
    (_, test_datasets), config = get_context_set(args, verbose=False)
    n_tasks            = len(test_datasets)
    classes_per_ctx    = config['classes_per_context']

    # model skeleton -----------------------------------------------------------
    model = define.define_classifier(
        args=args, config=config, device=device, depth=args.depth
    ).to(device).eval()
    model.scenario            = args.scenario
    model.classes_per_context = classes_per_ctx
    model.singlehead          = getattr(args, 'singlehead', False)

    print(f"\n==> Found {n_tasks} tasks; classes per task = {classes_per_ctx}\n")

    # loop over ctx‑checkpoints -----------------------------------------------
    for ctx_id in range(1, n_tasks + 1):
        ckpt_name = f"{model.name}-ctx{ctx_id}"
        utils.load_checkpoint(model, args.ckpt_dir, name=ckpt_name,
                              verbose=False, strict=False)

        task_accs = []
        for test_id in range(ctx_id):
            allowed = list(range(classes_per_ctx * test_id,
                                 classes_per_ctx * (test_id + 1)))
            acc, _ = evaluate.test_acc(
                model,
                test_datasets[test_id],
                verbose=False,
                context_id=test_id,
                allowed_classes=allowed,
            )
            task_accs.append(acc)
            print(f"CTX{ctx_id}  ->  Task {test_id + 1}: ACC = {acc:.4f}")

        avg_acc = np.mean(task_accs)
        print(f"CTX{ctx_id}  ->  Average over its {ctx_id} tasks: {avg_acc:.4f}\n")


if __name__ == "__main__":
    main()
