"""Eight list, dict and graph tasks for the benchmark. Five train, three holdout.

Same contract as tasks.py: every task has a reference solution, an obvious buggy solution
that fails a visible test, and a subtle buggy solution that passes every visible test but
fails a hidden one. Results are deterministic and never depend on set iteration order.
"""

from evorg.bench.model import Task

_DEDUPE = Task(
    id="dedupe_ordered",
    title="Remove duplicates, keep order",
    prompt=(
        "Return a new list with duplicate values removed, keeping the first occurrence of each "
        "value in its original order. Values are compared with ==, so 0 and False are equal. "
        "Items may be unhashable, such as lists. Empty input returns []."
    ),
    signature="def dedupe(items: list) -> list:",
    visible_tests=(
        "assert dedupe([2, 2, 1]) == [2, 1]",
        "assert dedupe([]) == []",
        "assert dedupe(['x']) == ['x']",
    ),
    hidden_tests=(
        "assert dedupe([1, 2, 1]) == [1, 2]",
        "assert dedupe([0, False]) == [0]",
        "assert dedupe([3, 3, 3]) == [3]",
        "assert dedupe([[1], [1]]) == [[1]]",
    ),
    reference_solution="""\
def dedupe(items: list) -> list:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out
""",
    buggy_solutions=(
        # Obvious: set() loses the original order, so [2, 2, 1] becomes [1, 2].
        """\
def dedupe(items: list) -> list:
    return list(set(items))
""",
        # Subtle: only drops a value equal to its immediate predecessor, so repeats that are
        # not adjacent survive. Every visible test uses adjacent repeats or none at all.
        """\
def dedupe(items: list) -> list:
    return [x for i, x in enumerate(items) if i == 0 or x != items[i - 1]]
""",
    ),
    difficulty=0.3,
    split="train",
)

_TRANSPOSE = Task(
    id="transpose_matrix",
    title="Transpose a matrix",
    prompt=(
        "Transpose a rectangular matrix given as a list of rows. The result's rows are the "
        "input's columns, so element [r][c] of the input becomes element [c][r] of the output. "
        "An empty matrix returns []. Return a new list and leave the input unchanged."
    ),
    signature="def transpose(matrix: list[list]) -> list[list]:",
    visible_tests=(
        "assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]",
        "assert transpose([[5]]) == [[5]]",
        "assert transpose([]) == []",
    ),
    hidden_tests=(
        "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]",
        "assert transpose([['a', 'b']]) == [['a'], ['b']]",
        "assert transpose([[1], [2], [3]]) == [[1, 2, 3]]",
        "assert transpose([[7, 8, 9, 10]]) == [[7], [8], [9], [10]]",
    ),
    reference_solution="""\
def transpose(matrix: list[list]) -> list[list]:
    return [list(col) for col in zip(*matrix)]
""",
    buggy_solutions=(
        # Obvious: returns the input rows unchanged.
        """\
def transpose(matrix: list[list]) -> list[list]:
    return [list(row) for row in matrix]
""",
        # Subtle: builds one output row per input row, which is right only for square
        # matrices. The visible matrices are all square or empty.
        """\
def transpose(matrix: list[list]) -> list[list]:
    return [[row[i] for row in matrix] for i in range(len(matrix))]
""",
    ),
    difficulty=0.4,
    split="train",
)

_ROTATE = Task(
    id="rotate_list",
    title="Rotate a list right",
    prompt=(
        "Rotate items to the right by k places: the last k elements move to the front, keeping "
        "their order, and the rest shift right. k is a non-negative integer and may be larger "
        "than the length, in which case it wraps around. Return a new list. Empty input "
        "returns []."
    ),
    signature="def rotate(items: list, k: int) -> list:",
    visible_tests=(
        "assert rotate([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]",
        "assert rotate([1, 2, 3, 4], 1) == [4, 1, 2, 3]",
        "assert rotate([], 3) == []",
        "assert rotate([1, 2, 3], 0) == [1, 2, 3]",
    ),
    hidden_tests=(
        "assert rotate([1, 2, 3, 4, 5], 7) == [4, 5, 1, 2, 3]",
        "assert rotate([1, 2], 5) == [2, 1]",
        "assert rotate([9], 100) == [9]",
        "assert rotate([1, 2, 3], 3) == [1, 2, 3]",
        "assert rotate([1, 2, 3, 4], 0) == [1, 2, 3, 4]",
    ),
    reference_solution="""\
def rotate(items: list, k: int) -> list:
    if not items:
        return []
    k %= len(items)
    cut = len(items) - k
    return items[cut:] + items[:cut]
""",
    buggy_solutions=(
        # Obvious: rotates left instead of right, so [1, 2, 3, 4, 5] by 2 gives [3, 4, 5, 1, 2].
        """\
def rotate(items: list, k: int) -> list:
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]
""",
        # Subtle: no modulo, so a k larger than the length slices past the list and returns
        # it unrotated. Every visible test has k no larger than the length.
        """\
def rotate(items: list, k: int) -> list:
    if not k:
        return list(items)
    return items[-k:] + items[:-k]
""",
    ),
    difficulty=0.5,
    split="train",
)

_GROUP_BY = Task(
    id="group_by_key",
    title="Group items by key",
    prompt=(
        "Group items by the value key(item), where key is a one-argument function. Return a "
        "dict mapping each key to the list of items with that key, in their original order. "
        "Dict keys appear in the order each key is first seen in items. Empty input returns {}."
    ),
    signature="def group_by(items: list, key) -> dict:",
    visible_tests=(
        "assert group_by(['apple', 'avocado', 'banana'], lambda s: s[0]) == "
        "{'a': ['apple', 'avocado'], 'b': ['banana']}",
        "assert group_by([1, 2, 3, 4], lambda x: x % 2) == {1: [1, 3], 0: [2, 4]}",
        "assert group_by([], len) == {}",
    ),
    hidden_tests=(
        "assert list(group_by([3, 1, 2], lambda x: x % 2).keys()) == [1, 0]",
        "assert group_by([1, 2, 3], lambda x: 'k') == {'k': [1, 2, 3]}",
        "assert list(group_by(['bb', 'a', 'cc'], len).keys()) == [2, 1]",
        "assert group_by(['bb', 'a', 'cc'], len) == {2: ['bb', 'cc'], 1: ['a']}",
        "assert group_by([(1, 'x'), (1, 'y')], lambda p: p[0]) == {1: [(1, 'x'), (1, 'y')]}",
    ),
    reference_solution="""\
def group_by(items: list, key) -> dict:
    out = {}
    for item in items:
        out.setdefault(key(item), []).append(item)
    return out
""",
    buggy_solutions=(
        # Obvious: a dict comprehension keeps only the last item for each key, so the
        # 'a' group becomes ['avocado'] instead of ['apple', 'avocado'].
        """\
def group_by(items: list, key) -> dict:
    return {key(item): [item] for item in items}
""",
        # Subtle: sorts the keys, so keys come out in sorted order instead of first-seen order.
        # Dict equality ignores order, so every visible test still passes.
        """\
def group_by(items: list, key) -> dict:
    out = {}
    for item in items:
        out.setdefault(key(item), []).append(item)
    return dict(sorted(out.items()))
""",
    ),
    difficulty=0.7,
    split="train",
)

_TWO_SUM = Task(
    id="two_sum_indices",
    title="Two sum indices",
    prompt=(
        "Find two different positions in nums whose values add up to target. Return the pair of "
        "indices (i, j) with i < j. Among all valid pairs return the one with the smallest i, "
        "and for that i the smallest j. The same position may not be used twice. Return None "
        "if no pair exists."
    ),
    signature="def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:",
    visible_tests=(
        "assert two_sum([2, 7, 11, 15], 9) == (0, 1)",
        "assert two_sum([3, 2, 4], 6) == (1, 2)",
        "assert two_sum([1, 2, 3], 100) is None",
    ),
    hidden_tests=(
        "assert two_sum([1, 2, 1, 2], 3) == (0, 1)",
        "assert two_sum([], 5) is None",
        "assert two_sum([5], 10) is None",
        "assert two_sum([-1, -2, 3], 1) == (1, 2)",
        "assert two_sum([0, 4, 3, 0], 0) == (0, 3)",
    ),
    reference_solution="""\
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None
""",
    buggy_solutions=(
        # Obvious: lets j start at 0, so an index can pair with itself. [3, 2, 4] by 6 returns
        # (0, 0) because 3 + 3 = 6.
        """\
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None
""",
        # Subtle: scans j from the end, so for the smallest i it returns the largest matching
        # j. Each visible test has only one valid pair for its first index.
        """\
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    for i in range(len(nums)):
        for j in range(len(nums) - 1, i, -1):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None
""",
    ),
    difficulty=0.9,
    split="train",
)

_WINDOW_MAX = Task(
    id="sliding_window_max",
    title="Sliding window maximum",
    prompt=(
        "Return the maximum of each contiguous window of length k in nums, in left-to-right "
        "order. The result has len(nums) - k + 1 items. Return [] if k is less than 1 or greater "
        "than len(nums). Values may repeat and may be negative."
    ),
    signature="def window_max(nums: list[int], k: int) -> list[int]:",
    visible_tests=(
        "assert window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]",
        "assert window_max([4, 4, 4], 1) == [4, 4, 4]",
        "assert window_max([1, 2], 3) == []",
    ),
    hidden_tests=(
        "assert window_max([9, 1, 1], 2) == [9, 1]",
        "assert window_max([], 3) == []",
        "assert window_max([5], 1) == [5]",
        "assert window_max([1, 3, 2], 0) == []",
        "assert window_max([1, 2, 3, 4], 4) == [4]",
    ),
    reference_solution="""\
from collections import deque


def window_max(nums: list[int], k: int) -> list[int]:
    if k < 1 or k > len(nums):
        return []
    out = []
    dq = deque()  # indices whose values are in decreasing order
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
""",
    buggy_solutions=(
        # Obvious: stops one window early, so the result is too short and misses the last
        # window. The standard example returns 5 values instead of 6.
        """\
def window_max(nums: list[int], k: int) -> list[int]:
    return [max(nums[i:i + k]) for i in range(len(nums) - k)]
""",
        # Subtle: the deque keeps an index that has just left the window, because it expires
        # indices with '<' instead of '<='. Stale maxima survive only when a window's
        # largest value sits at its left edge and the window then slides past it.
        """\
from collections import deque


def window_max(nums: list[int], k: int) -> list[int]:
    if k < 1 or k > len(nums):
        return []
    out = []
    dq = deque()
    for i, x in enumerate(nums):
        while dq and dq[0] < i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
""",
    ),
    difficulty=1.2,
    split="holdout",
)

_TOPO_SORT = Task(
    id="topo_sort",
    title="Topological sort with cycle detection",
    prompt=(
        "Given a directed graph as a dict mapping each node to the list of nodes that must come "
        "after it, return a topological order of all nodes, or None if the graph has a cycle. "
        "A node that appears only inside a successor list still counts as a node. Duplicate "
        "edges have no extra effect. When several nodes are ready, output the alphabetically "
        "smallest first. An empty graph returns []."
    ),
    signature="def topo_sort(graph: dict[str, list[str]]) -> list[str] | None:",
    visible_tests=(
        "assert topo_sort({'c': ['a'], 'a': [], 'b': []}) == ['b', 'c', 'a']",
        "assert topo_sort({'a': ['b'], 'b': ['a']}) is None",
        "assert topo_sort({}) == []",
        "assert topo_sort({'x': ['y', 'z'], 'y': ['z'], 'z': []}) == ['x', 'y', 'z']",
    ),
    hidden_tests=(
        "assert topo_sort({'a': ['c']}) == ['a', 'c']",
        "assert topo_sort({'a': ['b', 'b'], 'b': []}) == ['a', 'b']",
        "assert topo_sort({'a': ['a']}) is None",
        "assert topo_sort({'b': [], 'a': []}) == ['a', 'b']",
        "assert topo_sort({'a': ['b'], 'b': ['c'], 'c': ['a'], 'd': []}) is None",
    ),
    reference_solution="""\
def topo_sort(graph: dict[str, list[str]]) -> list[str] | None:
    nodes = set(graph)
    for succs in graph.values():
        nodes.update(succs)
    indeg = {n: 0 for n in nodes}
    for succs in graph.values():
        for s in succs:
            indeg[s] += 1
    ready = sorted(n for n in nodes if indeg[n] == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s in graph.get(n, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
        ready.sort()
    return order if len(order) == len(nodes) else None
""",
    buggy_solutions=(
        # Obvious: returns the sorted node names and ignores the edges, so the first visible
        # test gets ['a', 'b', 'c'] instead of ['b', 'c', 'a'].
        """\
def topo_sort(graph: dict[str, list[str]]) -> list[str] | None:
    return sorted(graph)
""",
        # Subtle: decrements in-degree once per distinct successor, but counted duplicate
        # edges when computing in-degree, so a duplicated edge never reaches zero. Visible
        # graphs have no duplicate edges.
        """\
def topo_sort(graph: dict[str, list[str]]) -> list[str] | None:
    nodes = set(graph)
    for succs in graph.values():
        nodes.update(succs)
    indeg = {n: 0 for n in nodes}
    for succs in graph.values():
        for s in succs:
            indeg[s] += 1
    ready = sorted(n for n in nodes if indeg[n] == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s in set(graph.get(n, [])):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
        ready.sort()
    return order if len(order) == len(nodes) else None
""",
    ),
    difficulty=1.6,
    split="holdout",
)

_GRID_SHORTEST = Task(
    id="grid_shortest_path",
    title="Shortest path on a grid",
    prompt=(
        "Find the fewest moves from S to G on a grid given as a list of equal-length strings. "
        "'#' is a wall and every other character is open floor. Each grid has exactly one S and "
        "one G. A move goes up, down, left or right by one cell and may not enter a wall or "
        "leave the grid. Return the number of moves, or -1 if G cannot be reached."
    ),
    signature="def shortest_path(grid: list[str]) -> int:",
    visible_tests=(
        "assert shortest_path(['S..', '...', '..G']) == 4",
        "assert shortest_path(['S#G']) == -1",
        "assert shortest_path(['S.#', '.##', '..G']) == 4",
    ),
    hidden_tests=(
        "assert shortest_path(['..G', '...', 'S..']) == 4",
        "assert shortest_path(['SG']) == 1",
        "assert shortest_path(['S.#G']) == -1",
        "assert shortest_path(['G#S', '..#', '...']) == -1",
        "assert shortest_path(['G..', '...', '..S']) == 4",
    ),
    reference_solution="""\
from collections import deque


def shortest_path(grid: list[str]) -> int:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        return -1
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#"
                    and (nr, nc) not in dist):
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return -1
""",
    buggy_solutions=(
        # Obvious: returns the Manhattan distance and ignores walls, so 'S#G' gives 2.
        """\
def shortest_path(grid: list[str]) -> int:
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                sr, sc = r, c
            elif ch == "G":
                gr, gc = r, c
    return abs(sr - gr) + abs(sc - gc)
""",
        # Subtle: searches only down and right, so any goal that lies up or to the left of
        # the start returns -1. Every visible goal lies down and to the right of S.
        """\
from collections import deque


def shortest_path(grid: list[str]) -> int:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        return -1
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#"
                    and (nr, nc) not in dist):
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return -1
""",
    ),
    difficulty=1.9,
    split="holdout",
)

DATA_TASKS: tuple[Task, ...] = (
    _DEDUPE,
    _TRANSPOSE,
    _ROTATE,
    _GROUP_BY,
    _TWO_SUM,
    _WINDOW_MAX,
    _TOPO_SORT,
    _GRID_SHORTEST,
)
