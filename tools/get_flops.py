# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
import os

from mmcv import Config, DictAction
from mmdet3d.models import build_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[40000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='image',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)

    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')

    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))
    # import thop
    # from thop import profile
    # from thop import clever_format
    # x=(3,384,1280)
    # MACs, params = profile(model, inputs=(x,))
    # print(f"运算量={MACs}, 参数量={params}")
    # # 将结果转换为更易于阅读的格式
    # MACs, params = clever_format([MACs, params], '%.3f')
    # print(f"运算量：{MACs}, 参数量：{params}")
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#             import thop
#             from thop import profile
#             from thop import clever_format
#             x = data['img_inputs'][0][0].to(device)
#             MACs, params = profile(model.to(device), inputs=(x,))
#             print(f"运算量={MACs}, 参数量={params}")
#             # 将结果转换为更易于阅读的格式
#             MACs, params = clever_format([MACs, params], '%.3f')
#             print(f"运算量：{MACs}, 参数量：{params}")

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#             import thop
#             from thop import profile
#             from thop import clever_format
#             x = data['img_inputs'][0][0].to(device)
#             data = dict_to_cuda(data, device=torch.device('cuda'))
#             model.to(device)
#             MACs, params = profile(model, inputs=(data,))
#             print(f"运算量={MACs}, 参数量={params}")
#             # 将结果转换为更易于阅读的格式
#             MACs, params = clever_format([MACs, params], '%.3f')
#             print(f"运算量：{MACs}, 参数量：{params}")

#             import thop
#             from thop import profile
#             from thop import clever_format
#             MACs, params = profile(model, inputs=(data,))
#             print(f"运算量={MACs}, 参数量={params}")
#             # 将结果转换为更易于阅读的格式
#             MACs, params = clever_format([MACs, params], '%.3f')
#             print(f"运算量：{MACs}, 参数量：{params}")

#
# def to_cuda(obj, device=None):
#     """递归将所有对象转移到CUDA（包括DataContainer和所有嵌套结构）"""
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     if isinstance(obj, torch.Tensor):
#         return obj.to(device)
#     elif isinstance(obj, DataContainer):
#         # 深度处理DataContainer内部所有数据
#         return DataContainer(
#             to_cuda(obj.data, device),
#             stack=obj.stack,
#             padding_value=obj.padding_value,
#             cpu_only=False  # 强制关闭cpu_only
#         )
#     elif isinstance(obj, dict):
#         return {k: to_cuda(v, device) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return type(obj)(to_cuda(item, device) for item in obj)
#     else:
#         # 非Tensor/DataContainer类型：尝试强制转换（如字符串等保持不变）
#         try:
#             return torch.tensor(obj).to(device)
#         except (TypeError, ValueError):
#             return obj  # 无法转换的类型保持原样
#
# def dict_to_cuda(data_dict, device=None):
#     """专为字典设计的CUDA转移（兼容原始需求）"""
#     return {k: to_cuda(v, device) for k, v in data_dict.items()}
