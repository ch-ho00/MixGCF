import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MixGCF")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon",
                        help="Choose a dataset:[amazon,yelp2018,ali]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    parser.add_argument("--ckpt", type=str, default=None, help="load ckpt")

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="ngcf",
                        help="Choose a recommender:[lightgcn, ngcf]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='concat', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=3, help="hop")
    parser.add_argument("--exp_num", type=int, default=0, help="Experiment Number based on ")
    parser.add_argument("--save_output", type=bool, default=False, help="use manifold mixup or not")
    #### new arguments
    # parser.add_argument("--expname", type=str ,required=True, help="experiment name")
    parser.add_argument('--alpha', type=float, default=0, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--alpha_gt', type=float, default=0, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--neg_mixup', type=bool, default=False, help='learning rate')
    parser.add_argument('--pos_mixup', type=bool, default=False, help='l2 regularization weight')
    parser.add_argument('--pos_neg_mixup', type=bool, default=False, help='learning rate')
    parser.add_argument('--lambda_mix', type=float, default=1e-4, help='lambda mix error loss weight')
    parser.add_argument('--lambda_fair', type=float, default=1e-4, help='lambda fair error loss weight')
    parser.add_argument('--fair', type=int, default=0, help='lambda mix error loss weight')
    parser.add_argument('--mixup', type=int, default=0, help='lambda fair error loss weight')
    parser.add_argument('--random_sample', type=float, default=0, help='lambda fair error loss weight')
    parser.add_argument('--expnum', type=str, default=0, help='lambda fair error loss weight')
    parser.add_argument('--root_dir', type=str, default='./experiments', help='lambda fair error loss weight')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument('--out_gpu', type=int, default=0, help='when using DP')
    parser.add_argument(
        "--out_dir", type=str, default="./weights/yelp2018/", help="output directory for model"
    )

    return parser.parse_args()
