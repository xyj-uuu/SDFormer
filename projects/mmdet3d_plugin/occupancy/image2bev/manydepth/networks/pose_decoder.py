# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
from collections import OrderedDict
# import torch
# import torch.nn as nn
#
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6):
        super(SimpleTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.linear = nn.Linear(d_model, 12)  

    def forward(self, x):
        # 
        batch_size, c, h, w = x.shape
        x = x.view(batch_size, c, -1)  
        x = x.permute(2, 0, 1)  
        x = self.transformer_encoder(x) 
        x = self.linear(x)  
        x = x.permute(1, 2, 0)  
        x = x.view(batch_size, 12, h, w)  
        return x

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.relu = nn.ReLU()

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.net = nn.ModuleList(list(self.convs.values()))


    
        self.transsim = SimpleTransformer()

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features
        out = self.transsim(out)

        out = self.relu(out)




        out = out.mean(3).mean(2)
        out = 0.01 * out.view( -1, self.num_frames_to_predict_for, 1, 6 )
        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
