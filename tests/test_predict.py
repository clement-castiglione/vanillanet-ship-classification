import numpy as np
import torch
from PIL import Image

from test_fusion import randomize_trained_state
from vanillanet_ships import VanillaNet7
from vanillanet_ships.predict import build_tta_transforms, load_model, predict_logits
from vanillanet_ships.train import TrainConfig, save_model


def test_tta_has_ten_views_of_the_right_shape():
    views = build_tta_transforms()
    assert len(views) == 10
    image = np.random.default_rng(0).integers(0, 256, (128, 192), dtype=np.uint8)
    for view in views:
        assert view(Image.fromarray(image)).shape == (1, 128, 192)


def test_fused_and_unfused_predictions_match(tmp_path):
    torch.manual_seed(0)
    cfg = TrainConfig(width=0.25)
    model = VanillaNet7(num_classes=10, width=cfg.width)
    randomize_trained_state(model)
    path = tmp_path / "model.pt"
    save_model(path, model, cfg, [f"class_{i}" for i in range(10)], epoch=0)

    images = np.random.default_rng(0).integers(0, 256, (6, 128, 192), dtype=np.uint8)
    views = build_tta_transforms()
    device = torch.device("cpu")
    fused, _ = load_model(path, device, fuse=True)
    unfused, _ = load_model(path, device, fuse=False)

    expected = predict_logits(unfused, images, views, device, batch_size=4)
    logits = predict_logits(fused, images, views, device, batch_size=4)
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4 * expected.abs().max().item())
    assert (logits.argmax(dim=1) == expected.argmax(dim=1)).all()
