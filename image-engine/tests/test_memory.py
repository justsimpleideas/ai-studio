import unittest

from image_engine.memory import (GIB, PLANS, TWO_EXPERT_PLANS, VIDEO_PLANS, choose_plan, choose_two_expert_plan,
                                 choose_video_plan, keep_both_experts)


class MemoryPlanTests(unittest.TestCase):
    def test_auto_picks_by_vram(self):
        cases = [(48, "full"), (24, "full"), (16, "quality"), (12, "balanced"), (11.94, "balanced"),
                 (8, "low"), (6, "low")]
        for gib, expected in cases:
            with self.subTest(gib=gib):
                self.assertEqual(choose_plan("auto", int(gib * GIB)).name, expected)

    def test_explicit_mode_wins_over_vram(self):
        self.assertEqual(choose_plan("low", 48 * GIB).name, "low")
        self.assertEqual(choose_plan("full", 8 * GIB).name, "full")

    def test_plans_are_consistent(self):
        self.assertFalse(PLANS["full"].swap_text_encoder or PLANS["full"].transformer_fp8)
        self.assertTrue(PLANS["balanced"].transformer_fp8 and PLANS["balanced"].swap_text_encoder)
        self.assertTrue(PLANS["low"].cpu_offload)


class VideoMemoryPlanTests(unittest.TestCase):
    def test_auto_picks_by_vram(self):
        # 11.94 GiB is about what a 12 GB card reports.
        cases = [(48, "full"), (32, "full"), (24, "quality"), (16, "balanced"), (11.94, "balanced"),
                 (8, "low"), (6, "low")]
        for gib, expected in cases:
            with self.subTest(gib=gib):
                self.assertEqual(choose_video_plan("auto", int(gib * GIB)).name, expected)

    def test_explicit_mode_wins_over_vram(self):
        self.assertEqual(choose_video_plan("full", 8 * GIB).name, "full")

    def test_only_the_largest_cards_keep_the_text_encoder_on_the_gpu(self):
        self.assertEqual([n for n, p in VIDEO_PLANS.items() if p.text_encoder_on_gpu], ["full"])

    def test_plans_are_consistent(self):
        for plan in VIDEO_PLANS.values():
            with self.subTest(plan=plan.name):
                # Parking the transformer only makes sense when it lives on the GPU in the first place.
                self.assertFalse(plan.park_transformer and plan.group_offload)
                self.assertFalse(plan.park_transformer and plan.vae_on_gpu)
        self.assertTrue(VIDEO_PLANS["balanced"].transformer_fp8)
        self.assertTrue(VIDEO_PLANS["low"].group_offload)


class TwoExpertPlanTests(unittest.TestCase):
    def test_a_12_gb_card_streams_the_experts(self):
        cases = [(48, "full"), (24, "quality"), (16, "balanced"), (11.94, "balanced"), (8, "balanced")]
        for gib, expected in cases:
            with self.subTest(gib=gib):
                self.assertEqual(choose_two_expert_plan("auto", int(gib * GIB)).name, expected)

    def test_both_experts_stay_loaded_only_with_room_for_them(self):
        expert, text = int(14.3 * GIB), int(11.4 * GIB)
        balanced, low = TWO_EXPERT_PLANS["balanced"], TWO_EXPERT_PLANS["low"]
        self.assertFalse(keep_both_experts(32 * GIB, expert, text, balanced))   # a 32 GB machine: one at a time
        self.assertTrue(keep_both_experts(64 * GIB, expert, text, balanced))
        self.assertFalse(keep_both_experts(64 * GIB, expert, text, low))        # Low VRAM never keeps both
        self.assertFalse(keep_both_experts(0, expert, text, balanced))          # RAM unknown


if __name__ == "__main__":
    unittest.main()
