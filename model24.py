import math
from opts import opt
from os.path import join
import torch
from torch import nn
import torch.nn.functional as F
from clip.model import CLIP, convert_weights
from einops import rearrange
from torchvision import transforms as T
from utils import *

def get_model_24(opt, name='Model'):
    model = eval(name)(opt)
    model.to("cuda")
    model = nn.DataParallel(model)
    return model

def gen_sineembed_for_position(pos_tensor, img_dim=1024):
    # bs, n_query, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 2048)
    scale = 2 * math.pi
    dim_t = torch.arange(img_dim // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (img_dim // 2))
    
    x_embed = pos_tensor[:, :, 0] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_x = rearrange(pos_x, 'b t c -> (b t) c')[None,:,:]
    
    y_embed = pos_tensor[:, :, 1] * scale
    pos_y = y_embed[:, :, None] / dim_t
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = rearrange(pos_y, 'b t c -> (b t) c')[None,:,:]
    
    # exp47
    """
    w_embed = pos_tensor[:, :, 2] * scale
    pos_w = w_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_w = rearrange(pos_w, 'b t c -> (b t) c')[None,:,:]

    h_embed = pos_tensor[:, :, 3] * scale
    pos_h = h_embed[:, :, None] / dim_t
    pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_h = rearrange(pos_h, 'b t c -> (b t) c')[None,:,:]

    pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=0)
    """
    x2_embed = (pos_tensor[:, :, 0] + pos_tensor[:, :, 2]) * scale
    pos_x2 = x2_embed[:, :, None] / dim_t
    pos_x2 = torch.stack((pos_x2[:, :, 0::2].sin(), pos_x2[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_x2 = rearrange(pos_x2, 'b t c -> (b t) c')[None,:,:]

    y2_embed = (pos_tensor[:, :, 1] + pos_tensor[:, :, 3]) * scale
    pos_y2 = y2_embed[:, :, None] / dim_t
    pos_y2 = torch.stack((pos_y2[:, :, 0::2].sin(), pos_y2[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y2 = rearrange(pos_y2, 'b t c -> (b t) c')[None,:,:]

    type_embed = pos_tensor[:, :, 4] * scale
    pos_type = type_embed[:, :, None] / dim_t
    pos_type = torch.stack((pos_type[:, :, 0::2].sin(), pos_type[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_type = rearrange(pos_type, 'b t c -> (b t) c')[None,:,:]

    pos = torch.cat((pos_x, pos_y, pos_x2, pos_y2, pos_type), dim=0)

    """
    c1_embed = ((pos_tensor[:, :, 0] + pos_tensor[:, :, 2]) / 2) * scale
    pos_c1 = c1_embed[:, :, None] / dim_t
    pos_c1 = torch.stack((pos_c1[:, :, 0::2].sin(), pos_c1[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_c1 = rearrange(pos_c1, 'b t c -> (b t) c')[None,:,:]

    c2_embed = ((pos_tensor[:, :, 1] + pos_tensor[:, :, 3]) / 2) * scale
    pos_c2 = c2_embed[:, :, None] / dim_t
    pos_c2 = torch.stack((pos_c2[:, :, 0::2].sin(), pos_c2[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_c2 = rearrange(pos_c2, 'b t c -> (b t) c')[None,:,:]

    pos = torch.cat((pos_y, pos_x, pos_c1, pos_c2), dim=0)
    """
    return pos

def xcorr_depthwise(x, kernel):
    """
    depthwise cross correlation
    ref: https://github.com/JudasDie/SOTS/blob/SOT/lib/models/sot/head.py#L227
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2))
    kernel = kernel.view(batch*channel, 1, kernel.size(2))
    out = F.conv1d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2))
    return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MyCLIP(CLIP):
    def __init__(self, *args):
        super(MyCLIP, self).__init__(*args)

    def encode_text_2(self, text, truncation=10):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        hidden = x[torch.arange(x.shape[0]), :truncation] @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return hidden, x

    def encode_text_(self, text):
        device = text.device
        B, L = text.size()  # L=77 (i.e., context_length)

        # original token/embedding
        token = text.detach()
        embedding = self.token_embedding(text).type(self.dtype).detach()

        # new token/embedding
        prompt_token = torch.zeros(B, 77)
        text_embedding = self.embedding(torch.arange(77).to(device))[None, :].repeat(B, 1, 1)  # [batch_size, n_ctx, d_model]

        # write token/embedding
        prefix, postfix = 4, 4
        for i in range(B):
            ind = torch.argmax(token[i], -1)  # EoT
            prompt_token[i, 0] = token[i, 0]
            prompt_token[i, prefix+1:prefix+ind] = token[i, 1:ind]
            prompt_token[i, prefix+ind+postfix] = token[i, ind]
            text_embedding[i, 0] = embedding[i,0]
            text_embedding[i, prefix+1: prefix+ind] = embedding[i, 1:ind]
            text_embedding[i, prefix+ind+postfix] = embedding[i, ind]
        prompt_token.to(device)
        text_embedding.to(device)
        x, text = text_embedding, prompt_token

        # copy from the original codes
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

def load_clip(model_path, input_resolution=None):
    state_dict = torch.jit.load(model_path).state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    if input_resolution is not None:
        if input_resolution != image_resolution:
            del state_dict['visual.attnpool.positional_embedding']
        image_resolution = input_resolution

    model = MyCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)

    return model

class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.mlp(x)
        x = x + self.drop(y)
        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.clip = load_clip(
            join(opt.save_root, f'CLIP/{opt.clip_model}.pt'),
            input_resolution=224,
        )
        self.clip = self.clip.float()
        self.dim = 1024
        self.img_fc = self.get_img_fc(use_ln=False)
        self.text_fc = self.get_text_fc(use_ln=False)
        self._freeze_text_encoder()

        self.pos_enc = nn.Sequential(
            MLP(self.dim*2, self.dim, self.dim, num_layers=2),
            nn.LayerNorm(self.dim, eps=1e-12),
        )

        if self.opt.kum_mode == 'cascade attention':
            self.fusion_visual_textual = nn.ModuleList([
                nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=4,
                dropout=0,
            ) for _ in range(opt.num_layers)])
            
            self.fusion_fc = nn.Linear(self.dim, self.dim)
            self.fusion_ffn = FFN(self.dim, 0.1)
            self.fusion_drop = nn.Dropout(p=0.1)
    
    def _freeze_text_encoder(self):
        """
        These parameters are not frozen:
        - list(self.clip.token_embedding.parameters())
        - [self.clip.positional_embedding]
        """
        for p in list(self.clip.transformer.parameters()) + \
                 list(self.clip.ln_final.parameters()):
        #        [self.clip.text_projection, ]:
            p.requires_grad = False

    def _init_weights_function(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, epoch=1e5):
        output = dict()
        textual_hidden, textual_feat = self.textual_encoding(x['exp'])

        if self.opt.kum_mode and (epoch >= self.opt.tg_epoch):
            fused_feat = self.visual_fuse(
                x['bbox'], textual_hidden, self.opt.kum_mode
            )
        else:
            fused_feat = self.visual_fuse(x['bbox'])
        logits = F.cosine_similarity(fused_feat, textual_feat)
        output['logits'] = logits
        output['vis_feat'] = fused_feat
        output['text_feat'] = textual_feat
        return output

    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [bt,c,l]->[bt,c]
        # temporal pooling
        if len(feat.shape) ==1:
            feat = feat[None,:]
        feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [b,c]
        # projection
        feat = self.img_fc(feat)
        return feat

    def cross_modal_fusion(self, vis_feat, text_feat, b, t):
        assert len(text_feat.size()) == 3
        # get textual embeddings
        text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
        text_feat = text_feat.repeat([1, t, 1, 1])
        text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')
        text_feat = self.fusion_fc(text_feat)
        text_feat = rearrange(text_feat, 'bt l c -> l bt c')
        # fusion
        fused_feat = vis_feat.clone()
        for layer in self.fusion_visual_textual:
            fused_feat = layer(
                query=fused_feat,
                key=text_feat,
                value=text_feat,
            )[0]
            fused_feat = self.fusion_ffn(fused_feat)
            fused_feat = self.fusion_drop(fused_feat)
            
        vis_feat = vis_feat * fused_feat
        vis_feat = rearrange(vis_feat, 'l bt c -> bt c l')
        return vis_feat

    def visual_fuse(self, bbox, text_feat=None, kum_mode=None):
        b, t = bbox.size()[:2]

        pos_feat = self.pos_enc(gen_sineembed_for_position(bbox, img_dim=4096))

        if kum_mode is not None:
            fused_feat = self.cross_modal_fusion(
                pos_feat, text_feat, b, t
            )
        else:
            fused_feat = rearrange(pos_feat, 'l bt c -> bt c l') # l = 4
        
        fused_feat = self.st_pooling(fused_feat, bs=b)

        if self.training:
            return fused_feat
        else:
            fused_feat = F.normalize(fused_feat, p=2, dim=-1)
            return fused_feat

    def textual_encoding(self, tokens):
        x_hidden, x = self.clip.encode_text_2(tokens, self.opt.truncation)
        x = self.text_fc(x)
        if self.training:
            return x_hidden, x
        else:
            return x_hidden, F.normalize(x, p=2, dim=-1)

    def get_img_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.dim, self.opt.feature_dim),
                nn.LayerNorm(self.opt.feature_dim, eps=1e-12),
            )
        else:
            return nn.Linear(self.dim, self.opt.feature_dim)

    def get_text_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.opt.feature_dim),
                nn.LayerNorm(self.opt.feature_dim, eps=1e-12),
            )
        else:
            return nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.opt.feature_dim),
            )

if __name__ == '__main__':
    pass