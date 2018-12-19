import argparse
import itertools
import os
import random
import sys
import json

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cs236605.train_results import FitResult
from . import models
from . import training

if os.name == 'nt':
    DATA_DIR = os.path.join(r"C:\Users\jonat\deeplearningcourse\cs236605-hw2", '.pytorch-datasets')
else:
    DATA_DIR = os.path.join(os.getenv('HOME'), '.pytorch-datasets')


def run_experiment(run_name, out_dir='./results', seed=None,
                   # Training params
                   bs_train=128, bs_test=None, batches=100, epochs=100,
                   early_stopping=3, checkpoints=None, lr=1e-3, reg=1e-3,
                   # Model params
                   filters_per_layer=[64], layers_per_block=2, pool_every=2,
                   hidden_dims=[1024], ycn=False,
                   **kw):
    """
    Execute a single run of experiment 1 with a single configuration.
    :param run_name: The name of the run and output file to create.
    :param out_dir: Where to write the output to.
    """
    if not seed:
        seed = random.randint(0, 2**31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select model class (experiment 1 or 2)
    model_cls = models.ConvClassifier if not ycn else models.YourCodeNet

    # TODO: Train
    # - Create model, loss, optimizer and trainer based on the parameters.
    #   Use the model you've implemented previously, cross entropy loss and
    #   any optimizer that you wish.
    # - Run training and save the FitResults in the fit_res variable.
    # - The fit results and all the experiment parameters will then be saved
    #  for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    x0, _ = ds_train[0]
    insize = x0.shape
    num_classes = 10

    model = model_cls(insize, num_classes, filters=filters_per_layer*layers_per_block, pool_every=pool_every, hidden_dims=hidden_dims)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg)
    trainer = training.TorchTrainer(model, loss_fn, optimizer, device=device)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs_train, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=bs_test, shuffle=False)
    fit_res = trainer.fit(dl_train, dl_test, epochs, early_stopping=early_stopping, print_every=1)
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)

    return fit_res.test_acc[-1],fit_res.train_acc[-1]



def save_experiment(run_name, out_dir, config, fit_res):
    output = dict(
        config=config,
        results=fit_res._asdict()
    )
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = FitResult(**output['results'])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description='CS236605 HW2 Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-exp', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument('--run-name', '-n', type=str,
                        help='Name of run and output file', required=True)
    sp_exp.add_argument('--out-dir', '-o', type=str, help='Output folder',
                        default='./results', required=False)
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=None, required=False)

    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches', type=int,
                        help='Number of batches per epoch', default=100)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of epochs', default=100)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without '
                             'improvement', default=3)
    sp_exp.add_argument('--checkpoints', type=int,
                        help='Save model checkpoints to this file when test '
                             'accuracy improves', default=None)
    sp_exp.add_argument('--lr', type=float,
                        help='Learning rate', default=1e-3)
    sp_exp.add_argument('--reg', type=float,
                        help='L2 regularization', default=1e-3)

    # # Model
    sp_exp.add_argument('--filters-per-layer', '-K', type=int, nargs='+',
                        help='Number of filters per conv layer in a block',
                        metavar='K', required=True)
    sp_exp.add_argument('--layers-per-block', '-L', type=int, metavar='L',
                        help='Number of layers in each block', required=True)
    sp_exp.add_argument('--pool-every', '-P', type=int, metavar='P',
                        help='Pool after this number of conv layers',
                        required=True)
    sp_exp.add_argument('--hidden-dims', '-H', type=int, nargs='+',
                        help='Output size of hidden linear layers',
                        metavar='H', required=True)
    sp_exp.add_argument('--ycn', action='store_true', default=False,
                        help='Whether to use your custom network')

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed

def cross():
    learn_rates=[1.0000e-05,1.0000e-04, 3.5112e-04, 1.2328e-03, 4.3288e-03, 1.5199e-02, 5.3367e-02,
        1.8738e-01, 6.5793e-01, 2.3101e+00, 8.1113e+00, 2.8480e+01, 1.0000e+02]
    batchsizes=[100,250,500,1000]
    regs=[1.0000e-04, 3.7276e-04, 1.3895e-03, 5.1795e-03, 1.9307e-02, 7.1969e-02,
        2.6827e-01, 1.0000e+00]
    filters=[32,64,128,256,512]
    layers=[1,2,4,6,8]
    pools=[2,3,4,5,6]
    hiddens=[56,128,256,512,1024]
    with open("./CROSS_VAL_RESULTS", 'a+') as f:
        for bs in batchsizes:
            for filt in filters:
                for L in layers:
                    for P in pools:
                        for H in hiddens:
                            for lr in learn_rates:
                                for reg in regs:
                                    try:
                                        run_name = "CR_VAL" + "_LR_" + str(lr) + "_REG_" + str(reg) + _BS_
                                        "+str(bs)+"
                                        _FILT_
                                        "+str(filt)+"
                                        _L_
                                        "+str(L)+"
                                        _P_
                                        "+str(P)+"
                                        _H_
                                        "+str(H)
                                        res = run_experiment(run_name, out_dir='./results', seed=None,
                                                             # Training params
                                                             bs_train=bs, bs_test=int(bs / 5), batches=100, epochs=100,
                                                             early_stopping=3, checkpoints=None, lr=lr, reg=reg,
                                                             # Model params
                                                             filters_per_layer=[filt], layers_per_block=L, pool_every=P,
                                                             hidden_dims=[H], ycn=False)
                                        print(run_name, "\nLast test acc ", res[0], "\nLast train acc ", res[1], "\n", file
                                              =f)
                                        print(
                                        "\n\n***********\n\n", run_name, "\nLast test acc ", res[0], "\nLast train acc ",
                                        res[1], "\n", "\n\n***********\n\n", file=open("./res", "a+"))
                                        print(
                                        "\n\n***********\n\n", run_name, "\nLast test acc ", res[0], "\nLast train acc ",
                                        res[1], "\n", "\n\n***********\n\n")
                                        f.write("\n\n***********\n\n", run_name, "\nLast test acc ", res[0],
                                                "\nLast train acc ", res[1], "\n", "\n\n***********\n\n")
                                    except:
                                        print("\n\n***********\n\n", run_name, " except \n", "\n\n***********\n\n")
                                        pass            

if __name__ == '__main__':
    # parsed_args = parse_cli()
    # subcmd_fn = parsed_args.subcmd_fn
    # del parsed_args.subcmd_fn
    # print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    print("Starting")
    cross()
