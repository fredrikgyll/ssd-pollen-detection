import torch

from utils import jaccard, intersect

"""
Testing basic computations.

All squares are 2x2
(0,0) is top left
(3,4) is bottom right

+-----+-----+┅┅┅┅┅+     +
| A1  ┊     |  A2 ┊
|     ┊     |     ┊
+═════+╌╌╌╌╌+╌╌╌╌╌+·····+
|     ┊     |     ┊     ·
|     ┊     |     ┊     ·
+-----+-----+┅┅┅┅┅+     +
║     ╎     ║     ╎     ·
║ B2  ╎     ║  B1 ╎  B3 ·
+═════+╌╌╌╌╌+╌╌╌╌╌+·····+
"""


def test_single_intersect():
    boxes_a = torch.tensor([[0, 0, 2, 2]])  # A1
    boxes_b = torch.tensor([[1, 1, 3, 3]])  # B1
    result = intersect(boxes_a, boxes_b)
    expexted = torch.tensor([[1.0]])
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_complete_intersect():
    boxes_a = torch.tensor([[0, 0, 2, 2]])  # A1
    boxes_b = torch.tensor([[0, 0, 2, 2]])  # A1
    result = intersect(boxes_a, boxes_b)
    expexted = torch.tensor([[4.0]])
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_no_intersect():
    boxes_a = torch.tensor([[0, 0, 2, 2]])  # A1
    boxes_b = torch.tensor([[1, 2, 3, 4]])  # B3
    result = intersect(boxes_a, boxes_b)
    expexted = torch.tensor([[0.0]])
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_multiple_intersect():
    boxes_a = torch.tensor(
        [
            [0, 0, 2, 2],  # A1
            [0, 1, 2, 3],  # A2
        ]
    )
    boxes_b = torch.tensor([[1, 1, 3, 3]])  # B1
    result = intersect(boxes_a, boxes_b)
    expexted = torch.tensor([[1.0], [2.0]])
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_multiple2_intersect():
    boxes_a = torch.tensor(
        [
            [0, 0, 2, 2],  # A1
            [0, 1, 2, 3],  # A2
        ]
    )
    boxes_b = torch.tensor(
        [
            [1, 1, 3, 3],  # B1
            [1, 0, 3, 2],  # B2
        ]
    )
    result = intersect(boxes_a, boxes_b)
    expexted = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_multiple3_intersect():
    boxes_a = torch.tensor(
        [
            [0, 0, 2, 2],  # A1
            [0, 1, 2, 3],  # A2
        ]
    )
    boxes_b = torch.tensor(
        [
            [1, 1, 3, 3],  # B1
            [1, 2, 3, 4],  # B3
        ]
    )
    result = intersect(boxes_a, boxes_b)
    expexted = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 1.0],
        ]
    )
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_single_jaccard():
    boxes_a = torch.tensor([[0, 0, 2, 2]])  # A1
    boxes_b = torch.tensor([[1, 1, 3, 3]])  # B1
    result = jaccard(boxes_a, boxes_b)
    expexted = torch.tensor([[1.0 / 7]])
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_multiple_jaccard():
    boxes_a = torch.tensor(
        [
            [0, 0, 2, 2],  # A1
            [0, 1, 2, 3],  # A2
        ]
    )
    boxes_b = torch.tensor([[1, 1, 3, 3]])  # B1
    result = jaccard(boxes_a, boxes_b)
    expexted = torch.tensor([[1.0 / 7], [2.0 / 6]])
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_multiple2_jaccard():
    boxes_a = torch.tensor(
        [
            [0, 0, 2, 2],  # A1
            [0, 1, 2, 3],  # A2
        ]
    )
    boxes_b = torch.tensor(
        [
            [1, 1, 3, 3],  # B1
            [1, 0, 3, 2],  # B2
        ]
    )
    result = jaccard(boxes_a, boxes_b)
    expexted = torch.tensor(
        [
            [1.0 / 7, 2.0 / 6],
            [2.0 / 6, 1.0 / 7],
        ]
    )
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)


def test_multiple3_jaccard():
    boxes_a = torch.tensor(
        [
            [0, 0, 2, 2],  # A1
            [0, 1, 2, 3],  # A2
        ]
    )
    boxes_b = torch.tensor(
        [
            [1, 1, 3, 3],  # B1
            [1, 2, 3, 4],  # B3
        ]
    )
    result = jaccard(boxes_a, boxes_b)
    expexted = torch.tensor(
        [
            [1.0 / 7, 0.0],
            [2.0 / 6, 1.0 / 7],
        ]
    )
    assert expexted.size() == result.size()
    assert torch.all(expexted == result)
