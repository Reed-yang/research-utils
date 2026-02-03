"""Image merging utility for combining adjacent image blocks into a single block."""

from mineru.utils.enum_class import ContentType


def _normalize_type(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _extract_image_body(block):
    for sub_block in block.get("blocks", []):
        if _normalize_type(sub_block.get("type")) == "image_body":
            return sub_block
    return None


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def are_adjacent(bbox1, bbox2, x_gap_threshold=30.0, y_gap_threshold=50.0):
    """
    Check if two bboxes are spatially adjacent (close enough to be merged).

    Two bboxes are considered adjacent if:
    - They overlap or are close in X direction AND
    - They overlap or are close in Y direction

    Args:
        bbox1: First bbox [x0, y0, x1, y1]
        bbox2: Second bbox [x0, y0, x1, y1]
        x_gap_threshold: Maximum horizontal gap between bboxes
        y_gap_threshold: Maximum vertical gap between bboxes

    Returns:
        True if the bboxes are adjacent
    """
    # Calculate gaps in X and Y directions
    x_gap = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
    y_gap = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))

    # Check if they are close enough in both directions
    return x_gap <= x_gap_threshold and y_gap <= y_gap_threshold


def merge_adjacent_images(
    img_body_blocks,
    img_caption_blocks,
    x_gap_threshold=30.0,
    y_gap_threshold=50.0,
    min_group_size=2,
):
    """
    Merge spatially adjacent images into single blocks using connected components.

    This function identifies groups of images that are close to each other
    (forming a grid or cluster) and merges them into single larger blocks.

    Args:
        img_body_blocks: List of image body blocks with 'bbox' and 'group_id' keys.
        img_caption_blocks: List of caption blocks with 'group_id' keys.
        x_gap_threshold: Maximum horizontal distance between images to be considered adjacent.
        y_gap_threshold: Maximum vertical distance between images to be considered adjacent.
        min_group_size: Minimum number of images in a group to trigger merging.

    Returns:
        Updated (img_body_blocks, img_caption_blocks) tuple.
    """
    if not img_body_blocks or len(img_body_blocks) < min_group_size:
        return img_body_blocks, img_caption_blocks

    n = len(img_body_blocks)
    uf = UnionFind(n)

    # Find all adjacent pairs and union them
    for i in range(n):
        bbox_i = img_body_blocks[i]["bbox"]
        for j in range(i + 1, n):
            bbox_j = img_body_blocks[j]["bbox"]
            if are_adjacent(bbox_i, bbox_j, x_gap_threshold, y_gap_threshold):
                uf.union(i, j)

    # Group blocks by their root
    from collections import defaultdict

    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)

    # Create merged blocks
    merged_blocks = []
    group_id_mapping = {}  # old_group_id -> new_group_id

    for root, indices in groups.items():
        if len(indices) == 1:
            # Single block, no merge needed
            merged_blocks.append(img_body_blocks[indices[0]])
        else:
            # Merge multiple blocks
            # Compute the bounding box that contains all blocks
            blocks_to_merge = [img_body_blocks[i] for i in indices]
            merged_bbox = [
                min(b["bbox"][0] for b in blocks_to_merge),
                min(b["bbox"][1] for b in blocks_to_merge),
                max(b["bbox"][2] for b in blocks_to_merge),
                max(b["bbox"][3] for b in blocks_to_merge),
            ]

            # Use the first block as template and update bbox
            merged_block = blocks_to_merge[0].copy()
            merged_block["bbox"] = merged_bbox

            # Map all old group_ids to the new group_id
            target_group_id = merged_block["group_id"]
            for block in blocks_to_merge:
                old_gid = block["group_id"]
                if old_gid != target_group_id:
                    group_id_mapping[old_gid] = target_group_id

            merged_blocks.append(merged_block)

    # Update caption group_ids
    for caption in img_caption_blocks:
        old_gid = caption["group_id"]
        if old_gid in group_id_mapping:
            caption["group_id"] = group_id_mapping[old_gid]

    return merged_blocks, img_caption_blocks


def merge_adjacent_image_blocks(
    image_blocks,
    x_gap_threshold=30.0,
    y_gap_threshold=50.0,
    min_group_size=2,
):
    """
    Merge two-layer image blocks based on adjacency of image bodies.

    Args:
        image_blocks: List of image blocks (type == "image") with nested "blocks".
        x_gap_threshold: Maximum horizontal distance between bodies to be considered adjacent.
        y_gap_threshold: Maximum vertical distance between bodies to be considered adjacent.
        min_group_size: Minimum number of images in a group to trigger merging.

    Returns:
        Updated image_blocks list.
    """
    if not image_blocks or len(image_blocks) < min_group_size:
        return image_blocks

    body_items = []
    for idx, image_block in enumerate(image_blocks):
        body = _extract_image_body(image_block)
        if body is None:
            continue
        body_items.append((idx, image_block, body))

    if len(body_items) < min_group_size:
        return image_blocks

    uf = UnionFind(len(body_items))
    for i in range(len(body_items)):
        bbox_i = body_items[i][2]["bbox"]
        for j in range(i + 1, len(body_items)):
            bbox_j = body_items[j][2]["bbox"]
            if are_adjacent(bbox_i, bbox_j, x_gap_threshold, y_gap_threshold):
                uf.union(i, j)

    from collections import defaultdict

    groups = defaultdict(list)
    for i in range(len(body_items)):
        groups[uf.find(i)].append(i)

    merged_blocks = []
    used_block_indices = set()

    for group in groups.values():
        if len(group) == 1:
            idx = body_items[group[0]][0]
            merged_blocks.append(image_blocks[idx])
            used_block_indices.add(idx)
            continue

        group_blocks = [body_items[i][1] for i in group]
        group_bodies = [body_items[i][2] for i in group]

        merged_bbox = [
            min(b["bbox"][0] for b in group_bodies),
            min(b["bbox"][1] for b in group_bodies),
            max(b["bbox"][2] for b in group_bodies),
            max(b["bbox"][3] for b in group_bodies),
        ]

        primary_body = min(group_bodies, key=lambda b: b.get("index", 0))
        merged_body = primary_body.copy()
        merged_body["bbox"] = merged_bbox
        merged_body["type"] = "image_body"

        merged_sub_blocks = []
        for block in group_blocks:
            for sub_block in block.get("blocks", []):
                if _normalize_type(sub_block.get("type")) == "image_body":
                    continue
                merged_sub_blocks.append(sub_block)

        merged_image_block = {
            "type": "image",
            "bbox": merged_bbox,
            "blocks": [merged_body],
            "index": merged_body.get("index", 0),
        }
        merged_image_block["blocks"].extend(merged_sub_blocks)
        merged_image_block["blocks"].sort(key=lambda x: x.get("index", 0))

        merged_blocks.append(merged_image_block)
        for i in group:
            used_block_indices.add(body_items[i][0])

    for idx, image_block in enumerate(image_blocks):
        if idx not in used_block_indices:
            merged_blocks.append(image_block)

    return merged_blocks


def merge_image_spans(spans, img_body_blocks, overlap_threshold=0.7):
    """
    Merge image spans that belong to the same merged image block.

    After img_body_blocks have been merged, this function updates the spans
    to reflect the merged blocks. Image spans that fall within a merged block's
    bbox are combined into a single span with the merged bbox.

    Args:
        spans: List of all spans (including image, text, etc.)
        img_body_blocks: List of (potentially merged) image body blocks
        overlap_threshold: Minimum overlap ratio for a span to be considered
                          part of a block (default 0.7)

    Returns:
        Updated spans list with merged image spans
    """
    if not spans or not img_body_blocks:
        return spans

    # Separate image spans from other spans
    image_spans = [s for s in spans if s.get("type") == ContentType.IMAGE]
    other_spans = [s for s in spans if s.get("type") != ContentType.IMAGE]

    if not image_spans:
        return spans

    # For each merged block, find which image spans belong to it
    # A span belongs to a block if it has significant overlap with the block bbox
    used_span_indices = set()
    new_image_spans = []

    for block in img_body_blocks:
        block_bbox = block["bbox"]
        block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])

        # Find all spans that have significant overlap with this block
        matching_spans = []
        for i, span in enumerate(image_spans):
            if i in used_span_indices:
                continue

            span_bbox = span["bbox"]

            # Calculate overlap
            x_overlap = max(0, min(block_bbox[2], span_bbox[2]) - max(block_bbox[0], span_bbox[0]))
            y_overlap = max(0, min(block_bbox[3], span_bbox[3]) - max(block_bbox[1], span_bbox[1]))
            overlap_area = x_overlap * y_overlap

            span_area = (span_bbox[2] - span_bbox[0]) * (span_bbox[3] - span_bbox[1])

            if span_area > 0 and overlap_area / span_area >= overlap_threshold:
                matching_spans.append((i, span))

        if len(matching_spans) > 1:
            # Multiple spans match this block - create a merged span
            merged_span = {
                "bbox": list(block_bbox),
                "type": ContentType.IMAGE,
            }
            scores = [s.get("score") for _, s in matching_spans if "score" in s]
            if scores:
                merged_span["score"] = max(scores)
            new_image_spans.append(merged_span)
            for i, _ in matching_spans:
                used_span_indices.add(i)
        elif len(matching_spans) == 1:
            # Single span - keep it as is
            i, span = matching_spans[0]
            new_image_spans.append(span)
            used_span_indices.add(i)

    # Add any remaining image spans that weren't matched
    for i, span in enumerate(image_spans):
        if i not in used_span_indices:
            new_image_spans.append(span)

    return other_spans + new_image_spans


def rebind_image_block_spans(image_blocks, spans, overlap_threshold=0.7):
    """
    Rebind image_body block lines to the merged image spans.

    Args:
        image_blocks: List of image blocks (type == "image") with nested "blocks".
        spans: List of all spans (including merged image spans).
        overlap_threshold: Minimum overlap ratio for a span to be bound to a block.
    """
    if not image_blocks or not spans:
        return

    image_spans = [s for s in spans if s.get("type") == ContentType.IMAGE]
    if not image_spans:
        return

    for block in image_blocks:
        body = _extract_image_body(block)
        if not body:
            continue
        block_bbox = body.get("bbox")
        if not block_bbox:
            continue

        best_span = None
        best_overlap = 0.0
        for span in image_spans:
            span_bbox = span.get("bbox")
            if not span_bbox:
                continue
            x_overlap = max(0, min(block_bbox[2], span_bbox[2]) - max(block_bbox[0], span_bbox[0]))
            y_overlap = max(0, min(block_bbox[3], span_bbox[3]) - max(block_bbox[1], span_bbox[1]))
            overlap_area = x_overlap * y_overlap
            span_area = (span_bbox[2] - span_bbox[0]) * (span_bbox[3] - span_bbox[1])
            if span_area <= 0:
                continue
            overlap_ratio = overlap_area / span_area
            if overlap_ratio >= overlap_threshold and overlap_ratio > best_overlap:
                best_span = span
                best_overlap = overlap_ratio

        if best_span:
            body["lines"] = [{"bbox": block_bbox, "spans": [best_span]}]
