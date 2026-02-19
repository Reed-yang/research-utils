import unittest
from mineru.utils.image_merge import merge_adjacent_images


class TestImageMerge(unittest.TestCase):
    def test_merge_grid_2x3(self):
        # Simulate a 2x3 grid of images
        # Row 1: (0,0)-(10,10), (12,0)-(22,10), (24,0)-(34,10)
        # Row 2: (0,12)-(10,22), (12,12)-(22,22), (24,12)-(34,22)
        # Assuming threshold > 2 and vertical overlap high.

        img_body_blocks = [
            {"bbox": [0, 0, 10, 10], "score": 1.0, "group_id": 0},
            {"bbox": [12, 0, 22, 10], "score": 1.0, "group_id": 1},
            {"bbox": [24, 0, 34, 10], "score": 1.0, "group_id": 2},
            {"bbox": [0, 12, 10, 22], "score": 1.0, "group_id": 3},
            {"bbox": [12, 12, 22, 22], "score": 1.0, "group_id": 4},
            {"bbox": [24, 12, 34, 22], "score": 1.0, "group_id": 5},
        ]

        # All have same caption for simplicity or no caption
        img_caption_blocks = []

        # Merge with sufficient X threshold (2)
        merged_blocks, _ = merge_adjacent_images(
            img_body_blocks, img_caption_blocks, x_threshold=5, y_iou_threshold=0.8
        )

        # Should merge row 1 into one block: [0, 0, 34, 10]
        # Should merge row 2 into one block: [0, 12, 34, 22]
        # Wait, my logic merges *horizontally adjacent*.
        # Does it merge rows together?
        # My logic:
        # Sort by X.
        # Loop: check distance & vertical overlap.
        # Row 1 items overlap in Y (IoU ~ 1.0).
        # Row 1 items are close in X (dist=2).
        # So Row 1 items should merge.
        # Row 2 items overlap in Y.
        # Row 2 items are close in X.
        # So Row 2 items should merge.
        # Row 1 and Row 2 do NOT overlap in Y.
        # So expectation: 2 blocks.

        self.assertEqual(len(merged_blocks), 2)

        # Verify bboxes
        # Note: Order might depend on sort.
        b1 = merged_blocks[0]["bbox"]
        b2 = merged_blocks[1]["bbox"]

        # One should be [0,0,34,10], other [0,12,34,22]

        bboxes = sorted([b1, b2], key=lambda x: x[1])
        self.assertEqual(bboxes[0], [0, 0, 34, 10])
        self.assertEqual(bboxes[1], [0, 12, 34, 22])

    def test_merge_single_row(self):
        img_body_blocks = [
            {"bbox": [0, 0, 10, 10], "score": 1.0, "group_id": 0},
            {"bbox": [100, 0, 110, 10], "score": 1.0, "group_id": 1},  # Far away
            {"bbox": [12, 0, 22, 10], "score": 1.0, "group_id": 2},  # Close to first
        ]
        # Sort order will be 0, 2, 1
        # 0 and 2 should merge. 1 should remain separate.

        merged_blocks, _ = merge_adjacent_images(img_body_blocks, [], x_threshold=5)
        self.assertEqual(len(merged_blocks), 2)

        # Merged block of 0 and 2
        block_merged = [b for b in merged_blocks if b["bbox"][0] == 0][0]
        self.assertEqual(block_merged["bbox"], [0, 0, 22, 10])

        # Single block 1
        block_single = [b for b in merged_blocks if b["bbox"][0] == 100][0]
        self.assertEqual(block_single["bbox"], [100, 0, 110, 10])


if __name__ == "__main__":
    unittest.main()
