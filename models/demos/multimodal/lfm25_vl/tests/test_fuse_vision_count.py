# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only check of the placeholder/feature count guard in the vision merge (ttnn is stubbed)."""

import importlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

_STUBBED = [
    "ttnn",
    "models.tt_transformers",
    "models.tt_transformers.tt",
    "models.tt_transformers.tt.model",
    "models.tt_transformers.tt.generator",
    "models.demos.multimodal.lfm25_vl.tt.decoder",
    "models.demos.multimodal.lfm25_vl.tt.vision_model",
]


@pytest.fixture(scope="module")
def e2e_cls():
    saved = {name: sys.modules.get(name) for name in _STUBBED + ["models.demos.multimodal.lfm25_vl.tt.e2e_model"]}
    try:
        for name in _STUBBED:
            sys.modules[name] = mock.MagicMock()
        # Real base classes so the model classes are created normally.
        sys.modules["models.tt_transformers.tt.model"].Transformer = type("Transformer", (), {})
        sys.modules["models.tt_transformers.tt.generator"].Generator = type("Generator", (), {})
        sys.modules.pop("models.demos.multimodal.lfm25_vl.tt.e2e_model", None)
        module = importlib.import_module("models.demos.multimodal.lfm25_vl.tt.e2e_model")
        for obj in vars(module).values():
            if isinstance(obj, type) and "_fuse_vision_into_text_embeddings" in vars(obj):
                yield obj
                return
        raise AssertionError("no class defines _fuse_vision_into_text_embeddings")
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _fuse(cls, tokens, features):
    self = SimpleNamespace(args=SimpleNamespace(image_token_index=7))
    embd = torch.zeros(1, tokens.shape[-1], 4)
    return cls._fuse_vision_into_text_embeddings(self, tokens, embd, features)


def test_matching_counts_are_spliced(e2e_cls):
    tokens = torch.tensor([[1, 7, 7, 2]])
    out = _fuse(e2e_cls, tokens, torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1)
    assert out[0, 1, 0] == 1 and out[0, 2, 0] == 5


@pytest.mark.parametrize("n_features", [1, 3])
def test_count_mismatch_raises(e2e_cls, n_features, expect_error):
    tokens = torch.tensor([[1, 7, 7, 2]])
    with expect_error(ValueError, "Image features and image tokens do not match"):
        _fuse(e2e_cls, tokens, torch.ones(n_features, 4))
