import os
import json

from utils.path import to_path


def save_arguments(output_path, args):
    output_path = to_path(output_path)

    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'arguments.json', 'w') as file:
        json.dump(args.__dict__, file, indent=4, sort_keys=True)


def print_arguments(args):
    print('*******************************************')
    print('**************** Arguments ****************')
    print('*******************************************\n')

    for arg, value in sorted(vars(args).items()):
        print(arg, '=', value)

    print(flush=True)
