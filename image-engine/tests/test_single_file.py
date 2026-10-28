import struct
import tempfile
import unittest
from pathlib import Path

from image_engine import single_file
from image_engine.single_file import SingleFileError
from tests.large_fixture import (PREFIX, TAG, TINY, TINY_6B, TINY_14B_TEXT, TINY_VIDEO, large_entries, six_b_config_json,
                                 six_b_entries, tiny_video_kinds, to_original_name, to_video_original, video_config_json,
                                 video_entries, write_safetensors)


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, fragment):
        with self.assertRaises(SingleFileError) as ctx:
            single_file.read_header(self.path)
        self.assertIn(fragment, str(ctx.exception))

    def test_a_well_formed_file_is_read(self):
        write_safetensors(self.path, {"a": ("F32", (2,), None), "b": ("U8", (3,), b"xyz")}, metadata={"k": "v"})
        header = single_file.read_header(self.path)
        self.assertEqual(header.tensors["a"].shape, (2,))
        self.assertEqual(single_file.read_bytes(self.path, header, "b"), b"xyz")
        self.assertEqual(header.metadata, {"k": "v"})

    def test_pickles_and_zips_are_named_as_such(self):
        self.path.write_bytes(b"PK\x03\x04" + bytes(100))
        self.refused("zip archive")
        self.path.write_bytes(b"\x80\x04" + bytes(100))
        self.refused("pickle")

    def test_bytes_after_the_last_tensor_are_refused(self):
        write_safetensors(self.path, {"a": ("F32", (2,), None)}, trailing=b"MZ hidden program")
        self.refused("belong to no tensor")

    def test_a_name_given_twice_is_refused(self):
        raw = b'{"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, ' \
              b'"a": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}'
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"xy")
        self.refused("names 'a' twice")

    def test_overlaps_and_wrong_lengths_are_refused(self):
        raw = b'{"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}, ' \
              b'"b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}}'
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"xyz")
        self.refused("gap or overlap")
        raw = b'{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}'
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"xyzw")
        self.refused("not as long as its shape")

    def test_metadata_must_be_strings(self):
        raw = b'{"__metadata__": {"k": 1}, "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}'
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"x")
        self.refused("__metadata__")

    def test_an_unknown_dtype_is_refused(self):
        raw = b'{"a": {"dtype": "OBJ", "shape": [1], "data_offsets": [0, 1]}}'
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"x")
        self.refused("unknown dtype")


class NameTests(unittest.TestCase):
    def test_every_name_round_trips_through_the_authors_layout(self):
        for name in single_file.large_expected_shapes(TINY):
            self.assertEqual(single_file.to_diffusers_name(to_original_name(name)), name)

    def test_names_as_published(self):
        cases = {
            "model.diffusion_model.blocks.0.attn.gate.weight": "transformer_blocks.0.attn.to_gate.weight",
            "model.diffusion_model.blocks.0.attn.qknorm.knorm.scale": "transformer_blocks.0.attn.norm_k.weight",
            "model.diffusion_model.blocks.3.mlp.down.weight": "transformer_blocks.3.ff.down.weight",
            "model.diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.weight": "text_fusion.layerwise_blocks.0.attn.to_out.0.weight",
            "model.diffusion_model.last.modulation.lin": "final_layer.scale_shift_table",
            "model.diffusion_model.tmlp.0.weight": "time_embed.linear_1.weight",
            "model.diffusion_model.first.bias": "img_in.bias",
        }
        for old, new in cases.items():
            self.assertEqual(single_file.to_diffusers_name(old, PREFIX), new)

    def test_diffusers_names_pass_through(self):
        for name in single_file.large_expected_shapes(TINY):
            self.assertEqual(single_file.to_diffusers_name(name), name)

    def test_a_prefix_on_only_some_names_is_refused(self):
        with self.assertRaises(SingleFileError):
            single_file.strip_prefix([PREFIX + "first.weight", "first.bias"])
        self.assertEqual(single_file.strip_prefix([PREFIX + "a", PREFIX + "b"]), PREFIX)
        self.assertEqual(single_file.strip_prefix(["a", "b"]), "")


class InspectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, entries):
        write_safetensors(self.path, entries)
        return self.path

    def refused(self, entries, fragment):
        self.write(entries)
        with self.assertRaises(SingleFileError) as ctx:
            single_file.inspect(self.path)
        self.assertIn(fragment, str(ctx.exception))

    def test_an_fp8_file_in_the_authors_layout(self):
        inspection = single_file.inspect(self.write(large_entries()))
        self.assertEqual(inspection.config, TINY)
        self.assertEqual(inspection.prefix, PREFIX)
        self.assertEqual(inspection.storage, "fp8")
        self.assertIn("transformer_blocks.1.ff.down.weight", inspection.quantized)
        self.assertEqual(inspection.quantized["transformer_blocks.1.ff.down.weight"].scale_shape, ())
        self.assertNotIn("img_in.weight", inspection.quantized)
        self.assertEqual(len(inspection.tags), len(inspection.quantized))
        self.assertEqual(inspection.names["final_layer.scale_shift_table"], PREFIX + "last.modulation.lin")
        self.assertEqual(inspection.parameters, sum(
            _numel(s) for s in single_file.large_expected_shapes(TINY).values()))

    def test_per_row_scales_bf16_files_and_diffusers_names(self):
        rows = single_file.inspect(self.write(large_entries(per_row=True)))
        self.assertEqual(rows.quantized["transformer_blocks.0.attn.to_q.weight"].scale_shape, (256,))
        plain = single_file.inspect(self.write(large_entries(fp8=False, prefix="")))
        self.assertEqual((plain.storage, plain.quantized, plain.prefix), ("BF16", {}, ""))
        diffusers = {name: ("BF16", shape, None) for name, shape in single_file.large_expected_shapes(TINY).items()}
        self.assertEqual(single_file.inspect(self.write(diffusers)).config, TINY)

    def test_the_published_dimensions_are_read_back(self):
        # The 12B model's own shapes, without writing 13 GB: only the header matters here.
        from image_engine.single_file import LargeConfig, infer_large_config, large_expected_shapes
        full = LargeConfig(64, 28, 128, 48, 12, 16384, 256, 2560, 12, 20, 20, 6912, 2, 2)
        shapes = large_expected_shapes(full)
        self.assertEqual(infer_large_config(shapes), full)
        self.assertAlmostEqual(sum(_numel(s) for s in shapes.values()) / 1e9, 12.6, delta=0.3)
        self.assertEqual(shapes["transformer_blocks.0.attn.to_k.weight"], (1536, 6144))

    def test_a_missing_or_extra_tensor_is_refused(self):
        entries = large_entries()
        del entries[PREFIX + "blocks.1.mlp.up.weight"]
        self.refused(entries, "missing")
        entries = large_entries()
        entries[PREFIX + "blocks.0.attn.extra.weight"] = ("BF16", (2, 2), None)
        self.refused(entries, "does not have")

    def test_a_wrong_shape_is_refused(self):
        entries = large_entries()
        entries[PREFIX + "blocks.1.mlp.up.weight"] = ("F8_E4M3", (64, 128), None)
        self.refused(entries, "expected [64, 256]")

    def test_fp8_rules(self):
        entries = large_entries()
        del entries[PREFIX + "blocks.0.mlp.gate.weight_scale"]
        self.refused(entries, "no weight_scale")
        entries = large_entries()
        entries[PREFIX + "first.weight"] = ("F8_E4M3", (256, 16), None)
        entries[PREFIX + "first.weight_scale"] = ("F32", (), None)
        self.write(entries)
        self.assertIn("img_in.weight", single_file.inspect(self.path).quantized)  # any linear weight may be fp8
        entries = large_entries()
        entries[PREFIX + "last.norm.scale"] = ("F8_E4M3", (256,), None)
        self.refused(entries, "only linear weights")
        entries = large_entries()
        entries[PREFIX + "blocks.0.mlp.gate.weight_scale"] = ("F32", (3,), None)
        self.refused(entries, "per tensor or per output row")
        entries = large_entries()
        entries[PREFIX + "blocks.0.mlp.gate.weight"] = ("BF16", (64, 256), None)
        self.refused(entries, "Scale without an fp8 weight")
        entries = large_entries()
        entries[PREFIX + "first.weight"] = ("I32", (256, 16), None)
        self.refused(entries, "stored as I32")

    def test_quantization_tags_are_read_and_checked(self):
        self.write(large_entries(tag=b'{"format": "float8_e4m3fn"}'))
        single_file.inspect(self.path)
        self.refused(large_entries(tag=b'{"format": "int4"}'), "names format 'int4'")
        self.refused(large_entries(tag=b'{"format": "float8_e4m3fn", "block_size": 32}'), "block_size")
        # As the published fp8 file tags its output projections (63 bytes): accepted, the engine widens every weight anyway.
        tag = b'{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}'
        self.assertEqual(len(tag), 63)
        self.write(large_entries(tag=tag))
        single_file.inspect(self.path)
        self.refused(large_entries(tag=b'{"format": "float8_e4m3fn", "full_precision_matrix_mult": "yes"}'), "true or false")
        self.refused(large_entries(tag=b"\x00\x01binary"), "not JSON")
        self.refused(large_entries(tag=b"[1, 2]"), "not a JSON object")
        self.refused(large_entries(tag=b"{" + b" " * 2000 + b"}"), "more than a tag should be")
        entries = large_entries()
        entries[PREFIX + "first.comfy_quant"] = ("U8", (len(TAG),), TAG)
        self.refused(entries, "belongs to no fp8 weight")

    def test_a_file_of_another_model_is_named_unsupported(self):
        write_safetensors(self.path, {"model.diffusion_model.layers.0.attention.qkv.weight": ("BF16", (6, 2), None)})
        with self.assertRaises(SingleFileError) as ctx:
            single_file.inspect(self.path)
        self.assertIn("not a model the engine supports", str(ctx.exception))


class SixBTests(unittest.TestCase):
    """The 6B model's single file: the authors' layout with fused q/k/v, or diffusers'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def inspect(self, entries):
        write_safetensors(self.path, entries)
        return single_file.inspect(self.path)

    def test_an_fp8_file_in_the_authors_layout(self):
        inspection = self.inspect(six_b_entries(extra_norm=True))
        self.assertEqual((inspection.family, inspection.config, inspection.storage), ("diffusion", TINY_6B, "fp8"))
        self.assertEqual(inspection.label, "6B image model")
        fused = PREFIX + "layers.1.attention.qkv.weight"
        self.assertEqual([inspection.names[f"layers.1.attention.to_{p}.weight"] for p in "qkv"], [fused] * 3)
        self.assertEqual([inspection.rows[f"layers.1.attention.to_{p}.weight"] for p in "qkv"], [(0, 64), (64, 128), (128, 192)])
        self.assertEqual(inspection.quantized["layers.1.attention.to_k.weight"].scale, fused + "_scale")
        self.assertEqual(inspection.names["all_final_layer.2-1.linear.weight"], PREFIX + "final_layer.linear.weight")
        self.assertEqual(inspection.names["layers.0.attention.to_out.0.weight"], PREFIX + "layers.0.attention.out.weight")
        self.assertEqual(inspection.ignored, [PREFIX + "norm_final.weight"])
        self.assertNotIn("cap_embedder.1.weight", inspection.quantized)

    def test_per_row_scales_of_a_fused_weight_are_cut_with_it(self):
        inspection = self.inspect(six_b_entries(per_row=True))
        quant = inspection.quantized["layers.0.attention.to_v.weight"]
        self.assertEqual((quant.scale_shape, quant.rows), ((64,), (128, 192)))

    def test_diffusers_names_and_bf16_files(self):
        inspection = self.inspect(six_b_entries(diffusers_names=True, fp8=False, prefix=""))
        self.assertEqual((inspection.config, inspection.storage, inspection.rows), (TINY_6B, "BF16", {}))

    def test_the_published_dimensions_are_read_back(self):
        from image_engine.single_file import SixBConfig, infer_six_b_config, six_b_expected_shapes
        full = SixBConfig(16, 3840, 30, 2, 30, 30, 2560)
        shapes = six_b_expected_shapes(full)
        self.assertEqual(infer_six_b_config(shapes), full)
        self.assertEqual(len(shapes), 521)  # every tensor of the downloaded model's transformer
        self.assertAlmostEqual(sum(_numel(s) for s in shapes.values()) / 1e9, 6.15, delta=0.01)

    def test_a_missing_tensor_or_a_wrong_config_is_refused(self):
        entries = six_b_entries()
        del entries[PREFIX + "layers.1.feed_forward.w3.weight"]
        write_safetensors(self.path, entries)
        with self.assertRaises(SingleFileError):
            single_file.inspect(self.path)
        inspection = self.inspect(six_b_entries())
        self.assertEqual(single_file.check_six_b_config(six_b_config_json(), inspection.config)["dim"], 64)
        with self.assertRaises(SingleFileError) as ctx:
            single_file.check_six_b_config({**six_b_config_json(), "n_layers": 30}, inspection.config)
        self.assertIn("n_layers", str(ctx.exception))

    def test_tags_must_belong_to_an_fp8_weight(self):
        entries = six_b_entries()
        entries[PREFIX + "cap_embedder.1.comfy_quant"] = ("U8", (len(TAG),), TAG)
        write_safetensors(self.path, entries)
        with self.assertRaises(SingleFileError):
            single_file.inspect(self.path)


class VideoTests(unittest.TestCase):
    """The video model's single file: the authors' layout or diffusers', bf16 or fp8."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def inspect(self, entries):
        write_safetensors(self.path, entries)
        return single_file.inspect(self.path)

    def test_every_name_round_trips_through_the_authors_layout(self):
        for name in single_file.video_expected_shapes(TINY_VIDEO):
            self.assertEqual(single_file.video_diffusers_name(to_video_original(name)), name)

    def test_a_bf16_file_in_the_authors_layout(self):
        inspection = self.inspect(video_entries())
        self.assertEqual((inspection.family, inspection.config, inspection.storage), ("video", TINY_VIDEO, "BF16"))
        self.assertEqual(inspection.names["blocks.0.norm2.weight"], PREFIX + "blocks.0.norm3.weight")
        self.assertEqual(inspection.names["blocks.1.attn2.to_out.0.weight"], PREFIX + "blocks.1.cross_attn.o.weight")
        self.assertEqual(inspection.names["scale_shift_table"], PREFIX + "head.modulation")
        self.assertEqual(inspection.names["blocks.0.scale_shift_table"], PREFIX + "blocks.0.modulation")
        self.assertEqual(inspection.names["condition_embedder.time_proj.weight"], PREFIX + "time_projection.1.weight")
        self.assertEqual(inspection.summary()["layers"], 2)

    def test_fp8_files_and_diffusers_names(self):
        inspection = self.inspect(video_entries(fp8=True, per_row=True))
        self.assertEqual(inspection.storage, "fp8")
        self.assertEqual(inspection.quantized["blocks.0.attn1.to_q.weight"].scale_shape, (32,))
        self.assertNotIn("proj_out.weight", inspection.quantized)
        self.assertEqual(len(inspection.tags), len(inspection.quantized))
        inspection = self.inspect(video_entries(diffusers_names=True, prefix=""))
        self.assertEqual((inspection.config, inspection.names["blocks.0.norm2.bias"]), (TINY_VIDEO, "blocks.0.norm2.bias"))

    def test_the_published_dimensions_are_read_back(self):
        from image_engine.single_file import VideoConfig, infer_video_config, video_expected_shapes
        five_b = VideoConfig(48, 48, 3072, 14336, 4096, 256, 30, (1, 2, 2))
        shapes = video_expected_shapes(five_b)
        self.assertEqual(infer_video_config(shapes), five_b)
        # The downloaded transformer's index lists 825 tensors, 19,999,150,848 bytes in fp32.
        self.assertEqual(len(shapes), 825)
        self.assertEqual(sum(_numel(s) for s in shapes.values()) * 4, 19_999_150_848)
        inspection = single_file.Inspection(family="video", config=five_b, prefix="", names={})
        self.assertEqual(inspection.label, "5B video model")

    def test_other_models_of_the_family_and_wrong_configs_are_refused(self):
        # An image-conditioned variant brings an image encoder this layout does not have.
        write_safetensors(self.path, video_entries(extra={"img_emb.proj.1.weight": (32, 32)}))
        with self.assertRaises(SingleFileError) as ctx:
            single_file.inspect(self.path)
        self.assertIn("does not match the model's structure", str(ctx.exception))
        inspection = self.inspect(video_entries())
        self.assertEqual(single_file.check_video_config(video_config_json(), inspection.config)["num_attention_heads"], 2)
        for change, fragment in (({"num_layers": 30}, "num_layers"), ({"num_attention_heads": 3}, "heads"),
                                 ({"image_dim": 1280}, "image_dim"), ({"patch_size": [1, 1, 1]}, "patch size")):
            with self.subTest(change=change), self.assertRaises(SingleFileError) as ctx:
                single_file.check_video_config({**video_config_json(), **change}, inspection.config)
            self.assertIn(fragment, str(ctx.exception))

    def test_the_older_fp8_layout_of_the_published_14b_files(self):
        inspection = self.inspect(video_entries(TINY_14B_TEXT, older_fp8=True, prefix=""))
        self.assertEqual((inspection.storage, inspection.video_kind), ("fp8", None))  # tiny: not a published size
        q = inspection.quantized["condition_embedder.time_embedder.linear_1.weight"]
        self.assertEqual((q.scale, q.scale_shape), ("time_embedding.0.scale_weight", ()))
        self.assertIn("blocks.0.attn1.to_q.weight", inspection.quantized)
        self.assertIn("scaled_fp8", inspection.ignored)
        self.assertIn("blocks.1.self_attn.q.scale_input", inspection.ignored)
        with tiny_video_kinds():
            self.assertEqual(self.inspect(video_entries(TINY_14B_TEXT, older_fp8=True, prefix="")).video_kind, "14b-text")

    def test_stray_input_scales_and_markers_are_refused(self):
        entries = video_entries(TINY_14B_TEXT, prefix="")
        entries["blocks.0.ffn.0.scale_input"] = ("F32", (), struct.pack("<f", 1.0))
        with self.assertRaises(SingleFileError) as ctx:
            self.inspect(entries)
        self.assertIn("Input scale without an fp8 weight", str(ctx.exception))
        entries = video_entries(TINY_14B_TEXT, prefix="")
        entries["scaled_fp8"] = ("F8_E4M3", (0,), None)
        with self.assertRaises(SingleFileError) as ctx:
            self.inspect(entries)
        self.assertIn("no fp8 weights", str(ctx.exception))

    def test_the_published_14b_sizes_are_known(self):
        from image_engine.single_file import VIDEO_KINDS, video_expected_shapes, video_kind
        self.assertEqual(video_kind(VIDEO_KINDS["14b-image"]), "14b-image")
        shapes = video_expected_shapes(VIDEO_KINDS["14b-text"])
        # The 14B download's transformer index: 1,095 tensors, 57,153,966,336 bytes in fp32.
        self.assertEqual(len(shapes), 1095)
        self.assertEqual(sum(_numel(s) for s in shapes.values()) * 4, 57_153_966_336)
        inspection = single_file.Inspection(family="video", config=VIDEO_KINDS["14b-image"], prefix="", names={})
        self.assertEqual(inspection.label, "14B image-to-video model (one expert of two)")

    def test_a_patch_embedding_in_fp8_is_refused(self):
        entries = video_entries()
        name = PREFIX + "patch_embedding.weight"
        entries[name] = ("F8_E4M3", entries[name][1], None)
        entries[PREFIX + "patch_embedding.weight_scale"] = ("F32", (), struct.pack("<f", 0.5))
        write_safetensors(self.path, entries)
        with self.assertRaises(SingleFileError) as ctx:
            single_file.inspect(self.path)
        self.assertIn("only linear weights", str(ctx.exception))


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


if __name__ == "__main__":
    unittest.main()
