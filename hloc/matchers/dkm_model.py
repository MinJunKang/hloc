
import torch

from torch import nn
from dkm.models.dkm import * 
from dkm.models.encoders import *
from kornia.geometry import resize


weight_urls = {
    "outdoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth",
    "indoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_indoor.pth",
}


class RegressionMatcher_DKM(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=384,
        w=512,
        use_contrastive_loss = False,
        alpha = 1,
        beta = 0,
        sample_mode = "threshold",
        upsample_preds = False,
        symmetric = False,
        name = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.use_contrastive_loss = use_contrastive_loss
        self.alpha = alpha
        self.beta = beta
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.symmetric = symmetric
        self.name = name
        self.sample_thresh = 0.05
        
    def extract_backbone_features(self, batch, batched = True):
        x_q = batch["query"]
        x_s = batch["support"]
        if batched:
            X = torch.cat((x_q, x_s))
            feature_pyramid = self.encoder(X)
        else:
            feature_pyramid = self.encoder(x_q), self.encoder(x_s)
        return feature_pyramid

    def sample(
        self,
        dense_matches,
        dense_certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            dense_certainty = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        elif "pow" in self.sample_mode:
            dense_certainty = dense_certainty**(1/3)
        elif "naive" in self.sample_mode:
            dense_certainty = torch.ones_like(dense_certainty)
        matches, certainty = (
            dense_matches.reshape(-1, 4).cpu().numpy(),
            dense_certainty.reshape(-1).cpu().numpy(),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = np.random.choice(
            np.arange(len(matches)),
            size=min(expansion_factor*num, len(certainty)),
            replace=False,
            p=certainty / np.sum(certainty),
        )
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        
        from dkm.utils.kde import kde
        density = kde(good_matches, std=0.1).cpu().numpy()
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        p = p/np.sum(p)
        balanced_samples = np.random.choice(
            np.arange(len(good_matches)),
            size=min(num,len(good_certainty)),
            replace=False,
            p = p,
        )
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, batched = True):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        if self.training and self.use_contrastive_loss:
            return dense_corresps, (f_q_pyramid, f_s_pyramid)
        else:
            return dense_corresps

    def forward_symmetric(self, batch):
        # save (feat_q, feat_s), (feat_s, feat_q) as pairs
        feature_pyramid = self.extract_backbone_features(batch)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]))
            for scale, f_scale in feature_pyramid.items()
        }
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        return dense_corresps

    def match(
        self,
        im1_torch,
        im2_torch,
        *args,
        batched=False,
    ):
        
        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            
            # inputs must be normalized and resized to (hs,ws)
            hw1 = im1_torch.shape[-2:]
            hw2 = im2_torch.shape[-2:]
            ws = self.w_resized
            hs = self.h_resized
            b = im1_torch.shape[0]
            assert(hw1 == hw2)
            if not batched:
                assert(hw1 == (hs, ws))
            batch = {"query": im1_torch, "support": im2_torch}
                
            finest_scale = 1
            # Run matcher
            if symmetric:
                dense_corresps  = self.forward_symmetric(batch)
            else:
                dense_corresps = self.forward(batch, batched = True)
            
            query_to_support = dense_corresps[finest_scale]["dense_flow"]
            # Get certainty interpolation
            dense_certainty = dense_corresps[finest_scale]["dense_certainty"]
            low_res_certainty = F.interpolate(
            dense_corresps[16]["dense_certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
            dense_certainty = dense_certainty - low_res_certainty
            query_to_support = query_to_support.permute(
                0, 2, 3, 1
                )
            
            if self.upsample_preds: 
                hs, ws = 864, 1152
                query = resize(im1_torch, (hs, ws), interpolation='bilinear')
                support = resize(im2_torch, (hs, ws), interpolation='bilinear')
                
                if symmetric:
                    query, support = torch.cat((query,support)), torch.cat((support,query))
                query_to_support, dense_certainty = self.decoder.upsample_preds(
                    query_to_support,
                    dense_certainty,
                    query,
                    support,
                )
            # Create im1 meshgrid
            query_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
                )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            dense_certainty = dense_certainty.sigmoid()  # logits -> probs
            query_coords = query_coords.permute(0, 2, 3, 1)
            if (query_to_support.abs() > 1).any() and True:
                wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
                dense_certainty[wrong[:,None]] = 0
            query_to_support = torch.clamp(query_to_support, -1, 1)
            if symmetric:
                qts, stq = query_to_support.chunk(2)
                q_warp = torch.cat((query_coords, qts), dim=-1)
                s_warp = torch.cat((stq, query_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp),dim=2)
                dense_certainty = torch.cat(dense_certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((query_coords, query_to_support), dim=-1)
            if batched:
                return (
                    warp,
                    dense_certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    dense_certainty[0, 0],
                )
                
                
def DKMv3(weights, h, w, symmetric = True, sample_mode= "threshold_balanced", **kwargs):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius = 3,
                corr_in_other = True,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius = 2,
                corr_in_other = True,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
            ),
            "1": ConvRefiner(
                2 * 3+6,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
        
    encoder = ResNet50(pretrained = False, high_res = False, freeze_bn=False)
    matcher = RegressionMatcher_DKM(encoder, decoder, h=h, w=w, name = "DKMv3", sample_mode=sample_mode, symmetric = symmetric, **kwargs)
    res = matcher.load_state_dict(weights)
    return matcher


def DKMv3_outdoor(path_to_weights = None, symmetric=True):
    """
    Loads DKMv3 outdoor weights, uses internal resolution of (540, 720) by default
    resolution can be changed by setting model.h_resized, model.w_resized later.
    Additionally upsamples preds to fixed resolution of (864, 1152),
    can be turned off by model.upsample_preds = False
    """
    if path_to_weights is not None:
        weights = torch.load(path_to_weights)
    else:
        weights = torch.hub.load_state_dict_from_url(weight_urls["outdoor"])
    return DKMv3(weights, 540, 720, symmetric=symmetric, upsample_preds = True)


def DKMv3_indoor(path_to_weights = None, symmetric=True):
    """
    Loads DKMv3 indoor weights, uses internal resolution of (480, 640) by default
    Resolution can be changed by setting model.h_resized, model.w_resized later.
    """
    if path_to_weights is not None:
        weights = torch.load(path_to_weights)
    else:
        weights = torch.hub.load_state_dict_from_url(weight_urls["indoor"])
    return DKMv3(weights, 480, 640, symmetric=symmetric, upsample_preds = False)