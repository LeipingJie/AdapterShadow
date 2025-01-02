import os
import torch
import torch.nn as nn

# downloading pretrained efficient model
def download_pretrained_efficientnet(name):
    os.environ['TORCH_HOME'] = './efficientnet'
    basemodel_name = f'tf_efficientnet_{name}_ap'
    print('Loading base model ()...'.format(basemodel_name), end='')
    #basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, source='local', pretrained=True)
    basemodel = torch.hub.load(repo_or_dir='./efficientnet/hub/rwightman_gen-efficientnet-pytorch_master', model=basemodel_name, source='local', pretrained=True)
    print('Done.')

    # Remove last layer
    print('Removing last two layers (global_pool & classifier).')
    basemodel.global_pool = nn.Identity()
    basemodel.classifier = nn.Identity()
    return basemodel

def get_efficientnet(name):
    return download_pretrained_efficientnet(name)