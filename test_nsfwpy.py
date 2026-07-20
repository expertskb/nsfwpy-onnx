import unittest
import numpy as np
from PIL import Image

import nsfwpy


class TestNSFWPy(unittest.TestCase):
    def test_package_import(self):
        """Verify package import."""
        self.assertIsNotNone(nsfwpy.__version__)

    def test_load_models(self):
        """Verify loading different ONNX model architectures."""
        model_default = nsfwpy.load_model()
        self.assertIsNotNone(model_default)

        model_vit = nsfwpy.load_model("nsfw_vit")
        self.assertIsNotNone(model_vit)

        model_quant = nsfwpy.load_model("nsfw_vit_quantized")
        self.assertIsNotNone(model_quant)


    def test_single_classification(self):
        """Verify single image classification return format."""
        model = nsfwpy.load_model("nsfw_vit_quantized")

        # Generate synthetic RGB PIL image
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        results = model.classify(img, top_k=5)

        self.assertGreater(len(results), 0)
        self.assertIn("className", results[0])
        self.assertIn("probability", results[0])

    def test_batch_classification(self):
        """Verify batch classification execution."""
        model = nsfwpy.load_model("nsfw_vit_quantized")
        img1 = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img2 = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

        results = model.classify_batch([img1, img2], top_k=3)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()

