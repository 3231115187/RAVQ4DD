from model import RAVQ4DD


def parse_method(args, n, c, d, device):

    num_types = getattr(args, 'real_num_node_types', 10)

    num_relations = getattr(args, 'real_num_relations', 4)

    model = RAVQ4DD(
        in_channels=d,
        hidden_channels=args.hidden_channels,
        out_channels=c,
        local_layers=args.local_layers,
        dropout=args.dropout,
        heads=args.num_heads,
        num_codes=args.num_codes,
        num_node_types=num_types,
        num_relations=num_relations,
        use_dynamic_codebook=args.use_dynamic_codebook
    ).to(device)
    return model


def parser_add_main_args(parser):
    # dataset
    parser.add_argument('--dataset', type=str, default='medical')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')


    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'])

    parser.add_argument('--train_prop', type=float, default=.6)
    parser.add_argument('--valid_prop', type=float, default=.2)
    parser.add_argument('--rand_split', action='store_true')
    parser.add_argument('--rand_split_class', action='store_true')

    parser.add_argument('--label_num_per_class', type=int, default=20)
    parser.add_argument('--valid_num', type=int, default=500)
    parser.add_argument('--test_num', type=int, default=1000)

    #模型参数
    parser.add_argument('--method', type=str, default='ravq4dd')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--local_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=2)

    #训练参数
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--in_dropout', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.6)

    #display
    parser.add_argument('--display_step', type=int, default=20)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument('--save_result', action='store_true', default=True)

    parser.add_argument('--kmeans', type=int, default=1)
    parser.add_argument('--num_codes', type=int, default=64)

    #Dynamic Codebook
    parser.add_argument('--use_dynamic_codebook', action='store_true', default=True)
    parser.add_argument('--aux_loss_weight', type=float, default=0.2)
    parser.add_argument('--gnn_loss_weight', type=float, default=0.5)

    parser.add_argument('--drop_edge_rate', type=float, default=0.25)
    parser.add_argument('--vq_noise_scale', type=float, default=0.01)

    #分析
    parser.add_argument('--analysis', action='store_true', default=True,
                        help='Run deep case study and heatmap generation')
    parser.add_argument('--load_pretrained', type=str, default=None)