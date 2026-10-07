<h2 align="center"> SCORE: Multimodal Representation Alignment Based
on Spherical Concentration [ICLR 2027 Submission]</a></h2>

<h3 align="center">
Anonymous Authors
</a></h3>


<h5 align="center"> 
<h5 align="center">



### ✨ Takeaway function

`circular_variance_computation`

SCORE measures the agreement of multiple normalized modality embeddings through their
**spherical concentration**.

Given a set of normalized embeddings, we compute the length of their mean resultant vector:

$$R = \left\|\frac{1}{K}\sum_{k=1}^{K} z_k\right\|_2$$

and the corresponding spherical variance:

$$
V = 1-R.
$$

When all modality embeddings point toward the same direction, the resultant vector is long
 $R \rightarrow 1$  and the variance is small  $V \rightarrow 0$ .
As the modalities become more dispersed, the resultant becomes shorter and the variance increases.

The following function computes this quantity for all cross-instance combinations in a batch:


```python
def circular_variance_computation(anchor, *inputs):
    """
    Compute spherical/circular variance for contrastive learning.

    Args:
        anchor (torch.Tensor):
            Tensor of shape (batch_size1, dim).

        *inputs (torch.Tensor):
            Variable number of modality tensors,
            each of shape (batch_size2, dim).

    Returns:
        torch.Tensor:
            Matrix of shape (batch_size1, batch_size2),
            where each entry contains the spherical variance
            of the corresponding multimodal tuple.
    """

    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Anchor-anchor dot products
    aa = torch.einsum(
        'bi,bi->b', anchor, anchor
    ).unsqueeze(1).expand(-1, batch_size2)

    # Pairwise dot products between the anchor and every other modality
    a_inputs = [
        anchor @ input.T
        for input in inputs
    ]

    # Dot products among non-anchor modalities
    input_dot_products = []

    for i, input1 in enumerate(inputs):
        row = []

        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum(
                'bi,bi->b', input1, input2
            ).unsqueeze(0).expand(batch_size1, -1)

            row.append(dot_product)

        input_dot_products.append(row)

    # Construct the Gram matrix for every candidate multimodal tuple
    G = torch.stack([
        torch.stack([aa] + a_inputs, dim=-1),
        *[
            torch.stack(
                [a_inputs[i]] + input_dot_products[i],
                dim=-1
            )
            for i in range(len(inputs))
        ]
    ], dim=-2)

    # Sum of Gram-matrix elements =
    # squared norm of the resultant vector
    gram_sum = torch.sum(
        G.float(),
        dim=(-2, -1)
    ).clamp(min=0.0)

    # Spherical variance:
    #
    # V = 1 - ||z_1 + ... + z_K||_2 / K
    #
    variance = (
        torch.ones_like(gram_sum)
        - torch.sqrt(gram_sum) / (1 + len(inputs))
    )

    return variance
```


The implementation uses the identity

$$
\left\|\sum_{k=1}^{K} z_k\right\|_2^2
=
\sum_{p=1}^{K}\sum_{q=1}^{K}
z_p^\top z_q,
$$

meaning that the squared norm of the resultant can be obtained by summing all the
entries of the Gram matrix.

Therefore, SCORE depends jointly on the agreement among **all modalities**, including
non-anchor modality pairs.


### 🧐 How to use it in practice?

Below is an implementation of the bidirectional InfoNCE loss using spherical variance
as the multimodal distance:


```python
import torch
import torch.nn.functional as F


# Hyperparameters
bs = 32
latent_dim = 512
contrastive_temp = 0.07


# Output of the encoders
language = torch.randn((bs, latent_dim))
video = torch.randn((bs, latent_dim))
audio = torch.randn((bs, latent_dim))


# Normalize embeddings onto the unit hypersphere
language = F.normalize(language, dim=-1)
video = F.normalize(video, dim=-1)
audio = F.normalize(audio, dim=-1)


# --------------------------------------------------
# Data-to-anchor direction
# --------------------------------------------------

variance = circular_variance_computation(
    language,
    video,
    audio
)

variance = variance / contrastive_temp


# --------------------------------------------------
# Anchor-to-data direction
# --------------------------------------------------

varianceT = circular_variance_computation(
    language,
    video,
    audio
).T

varianceT = varianceT / contrastive_temp


# Positive examples lie on the diagonal
targets = torch.arange(bs, dtype=torch.long)


# Lower spherical variance means stronger multimodal agreement,
# therefore we negate it before cross entropy.
loss = (
    F.cross_entropy(
        -variance,
        targets,
        label_smoothing=0.1
    )
    +
    F.cross_entropy(
        -varianceT,
        targets,
        label_smoothing=0.1
    )
) / 2


print(loss)
```


### 🔎 Why spherical concentration?

Traditional pairwise contrastive objectives generally align modalities using cosine
similarity between pairs of embeddings.

SCORE instead considers a complete multimodal tuple and measures how concentrated
its normalized embeddings are around a common direction.

For \(K\) normalized embeddings

$$
z_1,\ldots,z_K \in \mathbb{S}^{d-1},
$$

their mean resultant length is

$$
R =
\left\|
\frac{1}{K}
\sum_{k=1}^{K} z_k
\right\|_2.
$$

The corresponding spherical variance is

$$
V = 1-R.
$$

Hence:

- **perfect multimodal alignment** → \(R=1\), \(V=0\);
- **increasing multimodal disagreement** → smaller \(R\), larger \(V\).

Unlike objectives that only optimize anchor-to-modality relationships, the resultant
depends on all pairwise agreements inside the multimodal tuple. As a consequence,
optimizing spherical concentration promotes a common consensus direction across all
modalities.


### 🔄 Any number of modalities

The implementation accepts a variable number of non-anchor modalities:

```python
variance = circular_variance_computation(
    text,
    video,
    audio,
    subtitles
)
```

The exact same functional form can therefore be used with two, three, four, or more
available modalities.

For example:

```python
# Text + Video
variance_tv = circular_variance_computation(
    text,
    video
)

# Text + Audio
variance_ta = circular_variance_computation(
    text,
    audio
)

# Text + Video + Audio
variance_tva = circular_variance_computation(
    text,
    video,
    audio
)

# Text + Video + Audio + Subtitles
variance_tvas = circular_variance_computation(
    text,
    video,
    audio,
    subtitles
)
```

This makes the same learned representation space directly usable under
**variable-modality inference**.