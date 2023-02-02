import argparse
from common import utils


def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    # utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                            help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                            help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                            help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                            help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                            help='Training hidden size')
    enc_parser.add_argument('--skip', type=str,
                            help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                            help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                            help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                            help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                            help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                            help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                            help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                            help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                            help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                            help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                            help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
                            help='tag to identify the run')


    enc_parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
    enc_parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    # enc_parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    enc_parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    enc_parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    enc_parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    enc_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    enc_parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    enc_parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                        help='use mixed precision for training')
    enc_parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    enc_parser.add_argument('--data-dir',
                        help='location of the training dataset in the local filesystem (will be downloaded if needed)')

    # Arguments when not run through horovodrun
    enc_parser.add_argument('--num-proc', type=int)
    enc_parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
    enc_parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')


    enc_parser.set_defaults(conv_type='SAGE',
                            method_type='gnn',
                            dataset='scene',     # syn
                            n_layers=8,
                            batch_size=64,  # 64, batch 개수
                            hidden_dim=64,
                            skip="learnable",
                            dropout=0.0,
                            n_batches=10,  # 1000000, total 반복
                            opt='adam',     # opt_enc_parser
                            opt_scheduler='none',
                            opt_restart=10,
                            weight_decay=0.0,
                            lr=1e-4,
                            margin=0.1,
                            test_set='',
                            eval_interval=10,   # 1000, batch 반복횟수
                            n_workers=2,        # 4
                            model_path="ckpt/final/scene_model_ver3_10000_e1.pt",
                            tag='',
                            val_size=64,         # 4096,
                            node_anchored=False,    # True
                            cuda = "False",
                            num_proc = 1,
                            )

    # return enc_parser.parse_args(arg_str)
