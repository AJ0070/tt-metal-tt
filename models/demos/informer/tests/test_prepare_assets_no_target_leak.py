# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: the checkpoint quality gate must not feed the ground truth to the decoder."""

import torch

from models.demos.informer.reference.torch_reference import TorchInformerModel
from models.demos.informer.scripts.prepare_assets import build_default_etth1_config, evaluate_checkpoint_quality


class _RecordingModel(TorchInformerModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.future_values_args = []

    def forward(self, past_values, past_time_features, future_time_features, future_values=None):
        self.future_values_args.append(future_values)
        return super().forward(past_values, past_time_features, future_time_features, future_values)


def test_future_values_change_forward_output():
    torch.manual_seed(0)
    cfg = build_default_etth1_config()
    model = TorchInformerModel(cfg).eval()
    past = torch.randn(1, cfg.seq_len, cfg.enc_in)
    past_t = torch.randn(1, cfg.seq_len, cfg.time_feature_dim)
    future_t = torch.randn(1, cfg.pred_len, cfg.time_feature_dim)
    future = torch.randn(1, cfg.pred_len, cfg.enc_in)
    with torch.no_grad():
        out_zero = model(past, past_t, future_t, None)
        out_truth = model(past, past_t, future_t, future)
    assert not torch.allclose(out_zero, out_truth)


def test_quality_gate_passes_no_future_values():
    torch.manual_seed(0)
    cfg = build_default_etth1_config()
    model = _RecordingModel(cfg)
    n = 24 * 60 * 20 + cfg.seq_len + cfg.pred_len
    values = torch.randn(n, cfg.enc_in)
    time_features = torch.randn(n, cfg.time_feature_dim)
    evaluate_checkpoint_quality(model, values, time_features, cfg, min_improvement=-1e6, min_corr=-1.0)
    assert len(model.future_values_args) > 0
    assert all(arg is None for arg in model.future_values_args)
