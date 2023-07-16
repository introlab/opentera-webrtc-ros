import os
import json


def save_arguments(output_path, args):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'arguments.json'), 'w') as file:
        json.dump(args.__dict__, file, indent=4, sort_keys=True)


def print_arguments(args):
    print('*******************************************')
    print('**************** Arguments ****************')
    print('*******************************************\n')

    for arg, value in sorted(vars(args).items()):
        print(arg, '=', value)

    print(flush=True)
