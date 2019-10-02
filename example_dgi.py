import argparse
import random
import mlflow


def main():
    for epoch in range(random.randint(50, 150)):
        # log_metric logs metrics (acc, recall, ...) are for things that change
        mlflow.log_metric("precision", random.random() * epoch, step=epoch)
        mlflow.log_metric("recall", random.random() * epoch, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI')
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-3,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--runs", type=int, default=10,
                        help="number of total trainings")
    parser.add_argument("--n-dgi-epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--patience_reg", type=int, default=100,
                        help="early stop patience condition")
    parser.add_argument("--lamb", type=int, default=0.005,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--verbose", action='store_true',
                        help="print results")
    parser.add_argument("--MLI", action='store_true',
                        help="use multilayer infomax")

    parser.set_defaults(self_loop=True)
    parser.set_defaults(dataset="cora")
    parser.set_defaults(MLI=False)

    # MLFlow-specific arguments
    parser.add_argument("--experiment-name", type=str, default="Default",
                        help="name of mlflow experiment")
    parser.add_argument("--run-name", type=str, default=None,
                        help="name of mlflow run")
    parser.add_argument("--remote-server-uri", type=str, default="http://jarvis.babylontech.co.uk:5000/",
                        help="remote server URI")

    args = parser.parse_args()

    # MLFlow-specific logic
    mlflow.set_tracking_uri(args.remote_server_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_name=args.run_name)
    mlflow.log_params(vars(args))

    main()
