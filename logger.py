import torch
import os


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        # Data: [Train, Val, Test, Loss, Val_Micro, Val_Macro, Test_Micro, Test_Macro]
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = torch.tensor(self.results[run])
            if mode == 'max_acc':
                ind = result[:, 1].argmax().item()
            else:
                ind = result[:, 3].argmin().item()

            print(f'Run {run + 1:02d} Summary:')
            print(f'Best Epoch: {ind}')
            print(f'Train Acc: {100 * result[ind, 0]:.2f}%')
            print(f'Valid Acc: {100 * result[ind, 1]:.2f}%')
            print(f'Test  Acc: {100 * result[ind, 2]:.2f}%')
            print(f'Test  Micro-F1: {100 * result[ind, 6]:.2f}%')
            print(f'Test  Macro-F1: {100 * result[ind, 7]:.2f}%')

            self.test = result[ind, 2]
            return result[ind, 2]
        else:
            valid_results = [r for r in self.results if len(r) > 0]
            if len(valid_results) == 0: return torch.tensor(0.0)

            result = torch.tensor(valid_results)
            best_results = []
            for r in result:
                if mode == 'max_acc':
                    valid_idx = r[:, 1].argmax().item()
                else:
                    valid_idx = r[:, 3].argmin().item()

                final_train = r[valid_idx, 0].item()
                final_test = r[valid_idx, 2].item()
                final_micro = r[valid_idx, 6].item()
                final_macro = r[valid_idx, 7].item()
                best_results.append((final_train, final_test, final_micro, final_macro))

            best_result = torch.tensor(best_results)
            print(f'\nAll {len(best_result)} runs average:')
            print(f'Final Test Acc:      {100 * best_result[:, 1].mean():.2f} ± {100 * best_result[:, 1].std():.2f}')
            print(f'Final Test Micro-F1: {100 * best_result[:, 2].mean():.2f} ± {100 * best_result[:, 2].std():.2f}')
            print(f'Final Test Macro-F1: {100 * best_result[:, 3].mean():.2f} ± {100 * best_result[:, 3].std():.2f}')

            self.test = best_result[:, 1].mean()
            return best_result[:, 1]

    def output(self, out_path, info):
        with open(out_path, 'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')


def save_model(args, model, optimizer, run):
    if not os.path.exists(f'models/{args.dataset}'):
        os.makedirs(f'models/{args.dataset}')
    model_path = f'models/{args.dataset}/{args.method}_{run}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)


def load_model(args, model, optimizer, run):
    model_path = f'models/{args.dataset}/{args.method}_{run}.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method} {args.dropout} {args.lr} {results.mean():.2f} $\pm$ {results.std():.2f} \n")