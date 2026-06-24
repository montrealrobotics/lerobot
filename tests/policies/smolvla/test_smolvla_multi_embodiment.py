from types import SimpleNamespace

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    CategorySpecificLinear,
    SmolVLAPolicy,
)


def test_smolvla_prepare_embodiment_ids_prefers_explicit_id():
    config = SmolVLAConfig(use_category_specific_action_proj=True, max_num_embodiments=4)
    policy = object.__new__(SmolVLAPolicy)
    policy.config = config

    ids = policy.prepare_embodiment_ids(
        {
            "embodiment_id": torch.tensor([2, 3]),
            "dataset_index": torch.tensor([0, 1]),
        }
    )

    assert torch.equal(ids, torch.tensor([2, 3]))


def test_smolvla_prepare_embodiment_ids_falls_back_to_dataset_index():
    config = SmolVLAConfig(use_category_specific_action_proj=True, max_num_embodiments=4)
    policy = object.__new__(SmolVLAPolicy)
    policy.config = config

    ids = policy.prepare_embodiment_ids({"dataset_index": torch.tensor([0, 1])})

    assert torch.equal(ids, torch.tensor([0, 1]))


def test_smolvla_category_specific_linear_uses_per_batch_category():
    layer = CategorySpecificLinear(num_categories=2, input_dim=2, output_dim=1)
    with torch.no_grad():
        layer.W.zero_()
        layer.b.zero_()
        layer.W[0, :, 0] = torch.tensor([1.0, 0.0])
        layer.W[1, :, 0] = torch.tensor([0.0, 2.0])

    x = torch.tensor([[[3.0, 4.0]], [[3.0, 4.0]]])
    y = layer(x, torch.tensor([0, 1]))

    assert torch.allclose(y.squeeze(-1), torch.tensor([[3.0], [8.0]]))


def test_smolvla_dense_action_projection_weights_remap_to_pretrained_category():
    config = SmolVLAConfig(
        use_category_specific_action_proj=True,
        max_num_embodiments=3,
        max_action_dim=2,
        pretrained_action_proj_category=1,
    )
    policy = object.__new__(SmolVLAPolicy)
    policy.config = config
    policy.model = SimpleNamespace(
        action_in_proj=CategorySpecificLinear(3, 2, 4),
        action_out_proj=CategorySpecificLinear(3, 4, 2),
    )

    dense_in_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    dense_in_bias = torch.arange(4, dtype=torch.float32)
    dense_out_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dense_out_bias = torch.arange(2, dtype=torch.float32)

    remapped = policy._fix_category_specific_action_proj_state_dict(
        {
            "model.action_in_proj.weight": dense_in_weight,
            "model.action_in_proj.bias": dense_in_bias,
            "model.action_out_proj.weight": dense_out_weight,
            "model.action_out_proj.bias": dense_out_bias,
        }
    )

    assert "model.action_in_proj.weight" not in remapped
    assert "model.action_out_proj.weight" not in remapped
    assert torch.allclose(remapped["model.action_in_proj.W"][1], dense_in_weight.T)
    assert torch.allclose(remapped["model.action_in_proj.b"][1], dense_in_bias)
    assert torch.allclose(remapped["model.action_out_proj.W"][1], dense_out_weight.T)
    assert torch.allclose(remapped["model.action_out_proj.b"][1], dense_out_bias)
