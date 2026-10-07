# THIS IS THE CORE PY CODE OF SCORE FRAMEWORK
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityWeightNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
            #nn.Linear(hidden_dim, 3)
        )

        self._init_weights()
        #giving feat_t, feat_v and feat_a, compute a matrix of shape (batch_size1, batch_size2) where each cell (i,j) contains the concatenation of feat_t[i], feat_v[j] and feat_a[j]
    def concatenate_features(self, feat_t, feat_v, feat_a):
        batch_size1 = feat_t.shape[0]
        batch_size2 = feat_v.shape[0]
    
        # Expand feat_t to (batch_size1, batch_size2, dim)
        feat_t_expanded = feat_t.unsqueeze(1).expand(-1, batch_size2, -1)
    
        # Expand feat_v and feat_a to (batch_size1, batch_size2, dim)
        feat_v_expanded = feat_v.unsqueeze(0).expand(batch_size1, -1, -1)
        feat_a_expanded = feat_a.unsqueeze(0).expand(batch_size1, -1, -1)
    
        # Concatenate along the last dimension
        concatenated = torch.cat([feat_t_expanded, feat_v_expanded, feat_a_expanded], dim=-1)
    
        return concatenated

    def get_text_anchor_weights(self, feat_t, feat_v, feat_a, batch_size=128):
        b1 = feat_t.shape[0]
        all_w_t = []
        all_w_v = []
        all_w_a = []

        for i in range(0, b1, batch_size):
            chunk_t = feat_t[i:i + batch_size]
            x = self.concatenate_features(chunk_t, feat_v, feat_a)
            raw_w = self.mlp(x)
            w = F.softplus(raw_w) + 1e-6
            w_t_chunk, w_v_chunk, w_a_chunk = torch.ones_like(w[:,:, 0]).to(feat_t.device), w[:,:, 0], w[:,:, 1]
            all_w_t.append(w_t_chunk)
            all_w_v.append(w_v_chunk)
            all_w_a.append(w_a_chunk)

        w_t = torch.cat(all_w_t, dim=0)
        w_v = torch.cat(all_w_v, dim=0)
        w_a = torch.cat(all_w_a, dim=0)
        return w_t, w_v, w_a
    
    def get_vision_anchor_weights(self, feat_v, feat_a, feat_t, batch_size=128):
        b1 = feat_v.shape[0]
        all_w_v = []
        all_w_a = []
        all_w_t = []

        for i in range(0, b1, batch_size):
            chunk_v = feat_v[i:i + batch_size]
            x = self.concatenate_features(chunk_v, feat_a, feat_t)
            raw_w = self.mlp(x)
            w = F.softplus(raw_w) + 1e-6
            w_v_chunk, w_a_chunk, w_t_chunk = torch.ones_like(w[:,:, 0]).to(feat_v.device), w[:,:, 0], w[:,:, 1]
            all_w_v.append(w_v_chunk)
            all_w_a.append(w_a_chunk)
            all_w_t.append(w_t_chunk)

        w_v = torch.cat(all_w_v, dim=0)
        w_a = torch.cat(all_w_a, dim=0)
        w_t = torch.cat(all_w_t, dim=0)
        return w_v, w_a, w_t
    
    def get_audio_anchor_weights(self, feat_a, feat_t, feat_v, batch_size=128):
        b1 = feat_a.shape[0]
        all_w_a = []
        all_w_t = []
        all_w_v = []

        for i in range(0, b1, batch_size):
            chunk_a = feat_a[i:i + batch_size]
            x = self.concatenate_features(chunk_a, feat_t, feat_v)
            raw_w = self.mlp(x)
            w = F.softplus(raw_w) + 1e-6
            w_a_chunk, w_t_chunk, w_v_chunk = torch.ones_like(w[:,:, 0]).to(feat_a.device), w[:,:, 0], w[:,:, 1]
            all_w_a.append(w_a_chunk)
            all_w_t.append(w_t_chunk)
            all_w_v.append(w_v_chunk)

        w_a = torch.cat(all_w_a, dim=0)
        w_t = torch.cat(all_w_t, dim=0)
        w_v = torch.cat(all_w_v, dim=0)
        return w_a, w_t, w_v

    def forward(self, feat_t, feat_v, feat_a, batch_size=32):
        """
        matrix: (B1, B2 , D)
        returns:
            w_t, w_v, w_a: (B,)
        """
        # Process in mini-batches along the first dimension (B1) to avoid OOM
        b1 = feat_t.shape[0]
        all_w_t = []
        all_w_v = []
        all_w_a = []

        for i in range(0, b1, batch_size):
            chunk_t = feat_t[i:i + batch_size]
            x = self.concatenate_features(chunk_t, feat_v, feat_a)
            raw_w = self.mlp(x)

            w_t_chunk = torch.ones_like(raw_w[:,:, 0]).to(feat_t.device)
            w_v_chunk = (1.0 + raw_w[:,:, 0]).clamp(min=0.0, max=2.0)
            w_a_chunk = (1.0 + raw_w[:,:, 1]).clamp(min=0.0, max=2.0)

            all_w_t.append(w_t_chunk)
            all_w_v.append(w_v_chunk)
            all_w_a.append(w_a_chunk)

        w_t = torch.cat(all_w_t, dim=0)
        w_v = torch.cat(all_w_v, dim=0)
        w_a = torch.cat(all_w_a, dim=0)

        return w_t, w_v, w_a
    
    def _init_weights(self):
        """Initialize the last layer to output zeros"""
        # Initialize the last linear layer to output zeros
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

def circular_variance_computation(anchor, *inputs):
    """
    General function to compute circular variance for contrastive learning loss functions.
    Compute the circular variance metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for anchor with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for anchor with each input
    a_inputs = [anchor @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + a_inputs, dim=-1),
        *[torch.stack([a_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the summation for each Gram matrix, which is the squared norm of the mean vector
    gram_sum = torch.sum(G.float(), dim=(-2, -1)).clamp(min=0.0)

    # Compute the circular covariance
    res = torch.ones_like(gram_sum) - torch.sqrt(gram_sum) / (1 + len(inputs))
    return res


def weighted_circular_variance_computation(anchor, *inputs, weights):
    """
    Weighted circular variance computation.

    Args:
    - anchor (torch.Tensor): (batch_size1, dim)
    - *inputs (torch.Tensor): each (batch_size2, dim)
    - weights (list or tuple): weights [w_anchor, w_input1, ..., w_inputN]

    Returns:
    - torch.Tensor: (batch_size1, batch_size2)
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Number of modalities
    M = 1 + len(inputs)

    assert len(weights) == M, "weights must match number of modalities"

    w_anchor = weights[0]
    w_inputs = weights[1:]

    # --- anchor-anchor ---
    aa = (
        w_anchor ** 2
        * torch.einsum('bi,bi->b', anchor, anchor)
        .unsqueeze(1)
        .expand(-1, batch_size2)
    )

    # --- anchor-input ---
    a_inputs = [
        w_anchor * w_inputs[i] * (anchor @ input.T)
        for i, input in enumerate(inputs)
    ]

    # --- input-input ---
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = (
                w_inputs[i] * w_inputs[j]
                * torch.einsum('bi,bi->b', input1, input2)
                .unsqueeze(0)
                .expand(batch_size1, -1)
            )
            row.append(dot_product)
        input_dot_products.append(row)

    # --- weighted Gram matrix ---
    G = torch.stack(
        [
            torch.stack([aa] + a_inputs, dim=-1),
            *[
                torch.stack([a_inputs[i]] + input_dot_products[i], dim=-1)
                for i in range(len(inputs))
            ],
        ],
        dim=-2,
    )

    # --- squared norm of weighted resultant ---
    gram_sum = torch.sum(G.float(), dim=(-2, -1)).clamp(min=0.0)

    # --- normalization by sum of weights ---
    W = sum(weights)

    # --- weighted circular variance ---
    res = 1.0 - torch.sqrt(gram_sum) / W
    return res


def fully_dynamic_weighted_circular_variance(
    text, video, audio,
    w_t, w_v, w_a
):
    """
    Fully dynamic weighted circular variance (per-cell weights).

    Args:
    - text:  (B1, D)  text embeddings (normalized)
    - video: (B2, D)  video embeddings (normalized)
    - audio: (B2, D)  audio embeddings (normalized)
    - w_t:   (B1, B2) text weights
    - w_v:   (B1, B2) video weights
    - w_a:   (B1, B2) audio weights

    Returns:
    - cv: (B1, B2)
    """

    B1, D = text.shape
    B2 = video.shape[0]

    # ---- dot products ----
    tt = torch.einsum('bi,bi->b', text, text) \
             .unsqueeze(1).expand(-1, B2)

    tv = text @ video.T
    ta = text @ audio.T

    vv = torch.einsum('bi,bi->b', video, video) \
             .unsqueeze(0).expand(B1, -1)

    aa = torch.einsum('bi,bi->b', audio, audio) \
             .unsqueeze(0).expand(B1, -1)

    va = torch.einsum('bi,bi->b', video, audio) \
             .unsqueeze(0).expand(B1, -1)

    # ---- weighted squared norm ----
    gram_sum = (
        (w_t ** 2) * tt
        + (w_v ** 2) * vv
        + (w_a ** 2) * aa
        + 2 * w_t * w_v * tv
        + 2 * w_t * w_a * ta
        + 2 * w_v * w_a * va
    ).clamp(min=0.0)

    # ---- per-cell normalization ----
    W = (w_t + w_v + w_a).clamp(min=1e-8)

    cv = 1.0 - torch.sqrt(gram_sum) / W
    return cv




def fully_dynamic_weighted_circular_variance_4(
    text, video, audio, subtitles,
    w_t, w_v, w_a, w_s
):
    """
    Fully dynamic weighted circular variance (per-cell weights) for 4 modalities.

    Args:
    - text:       (B1, D) text embeddings (normalized)
    - video:      (B2, D) video embeddings (normalized)
    - audio:      (B2, D) audio embeddings (normalized)
    - subtitles:  (B2, D) subtitle embeddings (normalized)
    - w_t: (B1, B2) text weights
    - w_v: (B1, B2) video weights
    - w_a: (B1, B2) audio weights
    - w_s: (B1, B2) subtitle weights

    Returns:
    - cv: (B1, B2)
    """

    B1, D = text.shape
    B2 = video.shape[0]

    # ---- self dot products ----
    tt = torch.einsum('bi,bi->b', text, text) \
             .unsqueeze(1).expand(-1, B2)

    vv = torch.einsum('bi,bi->b', video, video) \
             .unsqueeze(0).expand(B1, -1)

    aa = torch.einsum('bi,bi->b', audio, audio) \
             .unsqueeze(0).expand(B1, -1)

    ss = torch.einsum('bi,bi->b', subtitles, subtitles) \
             .unsqueeze(0).expand(B1, -1)

    # ---- cross dot products ----
    tv = text @ video.T
    ta = text @ audio.T
    ts = text @ subtitles.T

    va = torch.einsum('bi,bi->b', video, audio) \
             .unsqueeze(0).expand(B1, -1)

    vs = torch.einsum('bi,bi->b', video, subtitles) \
             .unsqueeze(0).expand(B1, -1)

    a_s = torch.einsum('bi,bi->b', audio, subtitles) \
             .unsqueeze(0).expand(B1, -1)

    # ---- weighted squared norm expansion ----
    gram_sum = (
        (w_t ** 2) * tt
        + (w_v ** 2) * vv
        + (w_a ** 2) * aa
        + (w_s ** 2) * ss

        + 2 * w_t * w_v * tv
        + 2 * w_t * w_a * ta
        + 2 * w_t * w_s * ts

        + 2 * w_v * w_a * va
        + 2 * w_v * w_s * vs

        + 2 * w_a * w_s * a_s
    ).clamp(min=0.0)

    # ---- per-cell normalization ----
    W = (w_t + w_v + w_a + w_s).clamp(min=1e-8)

    cv = 1.0 - torch.sqrt(gram_sum) / W
    return cv