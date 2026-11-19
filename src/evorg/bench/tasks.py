"""The coding benchmark: thirty-six small Python tasks with visible and hidden tests.

Every task carries a reference solution, at least one obvious buggy solution (fails a
visible test) and at least one subtle buggy solution (passes every visible test but fails a
hidden one). Subtle bugs are what make a reviewer worth paying for.

Tests are single-line ``assert`` statements that call the function by name.
"""

from __future__ import annotations

from evorg.bench.model import SPLITS, Task
from evorg.bench.tasks_data import DATA_TASKS
from evorg.bench.tasks_numbers import NUMBER_TASKS
from evorg.bench.tasks_text import TEXT_TASKS

_FIZZBUZZ = Task(
    id="fizzbuzz_list",
    title="FizzBuzz list",
    prompt=(
        "Return a list of strings for the integers 1 through n inclusive. Use 'FizzBuzz' "
        "when the number is divisible by both 3 and 5, 'Fizz' when divisible by 3 only, "
        "'Buzz' when divisible by 5 only, and otherwise the number as a string. If n is "
        "less than 1 return an empty list."
    ),
    signature="def fizzbuzz(n: int) -> list[str]:",
    visible_tests=(
        "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
        "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
        "assert fizzbuzz(0) == []",
    ),
    hidden_tests=(
        "assert fizzbuzz(1) == ['1']",
        "assert fizzbuzz(-3) == []",
        "assert fizzbuzz(30)[14] == 'FizzBuzz'",
        "assert fizzbuzz(6)[2:] == ['Fizz', '4', 'Buzz', 'Fizz']",
        "assert fizzbuzz(10)[9] == 'Buzz'",
        "assert len(fizzbuzz(100)) == 100",
    ),
    reference_solution="""\
def fizzbuzz(n: int) -> list[str]:
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
""",
    buggy_solutions=(
        # Obvious: the FizzBuzz case is never reached, so 15 becomes 'Fizz'.
        """\
def fizzbuzz(n: int) -> list[str]:
    out = []
    for i in range(1, n + 1):
        if i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
""",
        # Subtle: negative input echoes the number instead of returning an empty list.
        """\
def fizzbuzz(n: int) -> list[str]:
    if n < 0:
        return [str(n)]
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
""",
    ),
    difficulty=0.2,
    split="train",
)

_REVERSE_WORDS = Task(
    id="reverse_words",
    title="Reverse words",
    prompt=(
        "Reverse the order of the words in s. Words are separated by any whitespace; the "
        "result joins them with single spaces and has no leading or trailing spaces. An "
        "empty or all-whitespace string returns ''. Do not reverse the characters within a "
        "word."
    ),
    signature="def reverse_words(s: str) -> str:",
    visible_tests=(
        "assert reverse_words('hello world') == 'world hello'",
        "assert reverse_words('a b c') == 'c b a'",
        "assert reverse_words('') == ''",
    ),
    hidden_tests=(
        "assert reverse_words('  one   two  ') == 'two one'",
        "assert reverse_words('solo') == 'solo'",
        "assert reverse_words('x\\ty\\nz') == 'z y x'",
        "assert reverse_words('   ') == ''",
        "assert reverse_words('the quick  brown fox') == 'fox brown quick the'",
    ),
    reference_solution="""\
def reverse_words(s: str) -> str:
    return " ".join(reversed(s.split()))
""",
    buggy_solutions=(
        # Obvious: reverses characters instead of words.
        """\
def reverse_words(s: str) -> str:
    return s[::-1]
""",
        # Subtle: splits on single spaces only, so tabs, newlines and runs of spaces break.
        """\
def reverse_words(s: str) -> str:
    return " ".join(reversed(s.split(" ")))
""",
    ),
    difficulty=0.4,
    split="train",
)

_IS_PALINDROME = Task(
    id="is_palindrome",
    title="Palindrome check",
    prompt=(
        "Return True if s reads the same forwards and backwards, ignoring letter case and "
        "every character that is not a letter or digit. The empty string and strings with "
        "no letters or digits are palindromes."
    ),
    signature="def is_palindrome(s: str) -> bool:",
    visible_tests=(
        "assert is_palindrome('racecar') is True",
        "assert is_palindrome('A man, a plan, a canal: Panama') is True",
        "assert is_palindrome('hello') is False",
    ),
    hidden_tests=(
        "assert is_palindrome('') is True",
        "assert is_palindrome('!!!') is True",
        "assert is_palindrome('a') is True",
        "assert is_palindrome(\"No 'x' in Nixon\") is True",
        "assert is_palindrome('ab') is False",
        "assert is_palindrome('0P') is False",
    ),
    reference_solution="""\
def is_palindrome(s: str) -> bool:
    kept = [c.lower() for c in s if c.isalnum()]
    return kept == kept[::-1]
""",
    buggy_solutions=(
        # Obvious: compares the raw string, so punctuation and case break it.
        """\
def is_palindrome(s: str) -> bool:
    return s == s[::-1]
""",
        # Subtle: strips only space, comma and colon, so apostrophes stay in the comparison.
        """\
def is_palindrome(s: str) -> bool:
    kept = "".join(c.lower() for c in s if c not in " ,:")
    return kept == kept[::-1]
""",
    ),
    difficulty=0.6,
    split="train",
)

_RLE_ENCODE = Task(
    id="rle_encode",
    title="Run-length encoding",
    prompt=(
        "Run-length encode s. Each maximal run of identical characters becomes the run length "
        "in decimal followed by the character, so 'aaabcc' becomes '3a1b2c'. Counts of ten or "
        "more are written in full, so twelve 'a' become '12a'. The empty string encodes to "
        "the empty string. Comparison is case-sensitive."
    ),
    signature="def rle_encode(s: str) -> str:",
    visible_tests=(
        "assert rle_encode('aaabcc') == '3a1b2c'",
        "assert rle_encode('') == ''",
        "assert rle_encode('abc') == '1a1b1c'",
    ),
    hidden_tests=(
        "assert rle_encode('a') == '1a'",
        "assert rle_encode('a' * 12) == '12a'",
        "assert rle_encode('x' * 10) == '10x'",
        "assert rle_encode('aabbaa') == '2a2b2a'",
        "assert rle_encode('AAaa') == '2A2a'",
        "assert rle_encode('1111') == '41'",
    ),
    reference_solution="""\
def rle_encode(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{j - i}{s[i]}")
        i = j
    return "".join(out)
""",
    buggy_solutions=(
        # Obvious: returns the input unchanged.
        """\
def rle_encode(s: str) -> str:
    return s
""",
        # Subtle: run lengths wrap modulo ten, so runs of ten or more lose digits.
        """\
def rle_encode(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{(j - i) % 10}{s[i]}")
        i = j
    return "".join(out)
""",
    ),
    difficulty=0.9,
    split="train",
)

_ROMAN_TO_INT = Task(
    id="roman_to_int",
    title="Roman numeral to integer",
    prompt=(
        "Convert a Roman numeral written in uppercase letters I, V, X, L, C, D and M to its "
        "integer value. A smaller symbol before a larger one is subtracted (IV is 4, IX is 9, "
        "XL is 40, XC is 90, CD is 400, CM is 900). The input is a valid numeral for 1 to "
        "3999. The empty string converts to 0."
    ),
    signature="def roman_to_int(s: str) -> int:",
    visible_tests=(
        "assert roman_to_int('III') == 3",
        "assert roman_to_int('IV') == 4",
        "assert roman_to_int('MCMXCIV') == 1994",
    ),
    hidden_tests=(
        "assert roman_to_int('I') == 1",
        "assert roman_to_int('LVIII') == 58",
        "assert roman_to_int('MMMCMXCIX') == 3999",
        "assert roman_to_int('XL') == 40",
        "assert roman_to_int('CDXLIV') == 444",
        "assert roman_to_int('') == 0",
    ),
    reference_solution="""\
def roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        if i + 1 < len(s) and v < values[s[i + 1]]:
            total -= v
        else:
            total += v
    return total
""",
    buggy_solutions=(
        # Obvious: never subtracts, so IV becomes 6.
        """\
def roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(values[ch] for ch in s)
""",
        # Subtle: correct for every non-empty numeral, but indexes s[0] unconditionally.
        """\
def roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = values[s[0]]
    for i in range(1, len(s)):
        prev, cur = values[s[i - 1]], values[s[i]]
        total += cur - 2 * prev if prev < cur else cur
    return total
""",
    ),
    difficulty=1.1,
    split="train",
)

_MERGE_INTERVALS = Task(
    id="merge_intervals",
    title="Merge intervals",
    prompt=(
        "Given a list of closed integer intervals (start, end) with start <= end, return the "
        "merged intervals sorted by start. Intervals that overlap or touch (one ends exactly "
        "where the next starts) are merged. The input may be unsorted and may contain "
        "duplicates. Return [] for an empty list."
    ),
    signature="def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:",
    visible_tests=(
        "assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == "
        "[(1, 6), (8, 10), (15, 18)]",
        "assert merge_intervals([(1, 4), (4, 5)]) == [(1, 5)]",
        "assert merge_intervals([]) == []",
    ),
    hidden_tests=(
        "assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]",
        "assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]",
        "assert merge_intervals([(1, 2)]) == [(1, 2)]",
        "assert merge_intervals([(2, 2), (2, 2)]) == [(2, 2)]",
        "assert merge_intervals([(3, 4), (1, 2), (2, 3)]) == [(1, 4)]",
    ),
    reference_solution="""\
def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            prev_start, prev_end = result[-1]
            result[-1] = (prev_start, max(prev_end, end))
        else:
            result.append((start, end))
    return result
""",
    buggy_solutions=(
        # Obvious: sorts but never merges.
        """\
def merge_intervals(intervals):
    return sorted(intervals)
""",
        # Subtle: a single pass that assumes the input is already sorted by start.
        """\
def merge_intervals(intervals):
    result = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            prev_start, prev_end = result[-1]
            result[-1] = (prev_start, max(prev_end, end))
        else:
            result.append((start, end))
    return result
""",
    ),
    difficulty=1.4,
    split="train",
)

_BALANCED_BRACKETS = Task(
    id="balanced_brackets",
    title="Balanced brackets",
    prompt=(
        "Return True if every opening bracket '(', '[' or '{' in s is closed by the matching "
        "closing bracket in the correct nesting order, and no closing bracket is unmatched. "
        "All other characters are ignored. The empty string is balanced."
    ),
    signature="def is_balanced(s: str) -> bool:",
    visible_tests=(
        "assert is_balanced('()[]{}') is True",
        "assert is_balanced('([)]') is False",
        "assert is_balanced('a(b)c') is True",
    ),
    hidden_tests=(
        "assert is_balanced('') is True",
        "assert is_balanced('(') is False",
        "assert is_balanced(')(') is False",
        "assert is_balanced('{[()()]}') is True",
        "assert is_balanced('((]') is False",
    ),
    reference_solution="""\
def is_balanced(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
""",
    buggy_solutions=(
        # Obvious: compares bracket counts, ignoring order, so '([)]' passes.
        """\
def is_balanced(s: str) -> bool:
    return all(s.count(o) == s.count(c) for o, c in ("()", "[]", "{}"))
""",
        # Subtle: never checks that the stack is empty at the end, so open brackets pass.
        """\
def is_balanced(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return True
""",
    ),
    difficulty=1.2,
    split="train",
)

_SECOND_LARGEST = Task(
    id="second_largest_distinct",
    title="Second largest distinct value",
    prompt=(
        "Return the second largest distinct value in nums, or None if nums has fewer than two "
        "distinct values. Duplicates of the largest value do not count as a second value. "
        "Negative numbers are allowed. An empty list returns None."
    ),
    signature="def second_largest(nums: list[int]) -> int | None:",
    visible_tests=(
        "assert second_largest([3, 1, 4, 1, 5]) == 4",
        "assert second_largest([7, 7, 7]) is None",
        "assert second_largest([2, 1]) == 1",
    ),
    hidden_tests=(
        "assert second_largest([]) is None",
        "assert second_largest([5]) is None",
        "assert second_largest([-1, -2, -3]) == -2",
        "assert second_largest([10, 10, 9]) == 9",
        "assert second_largest([1, 2, 3, 4, 4, 4]) == 3",
    ),
    reference_solution="""\
def second_largest(nums: list[int]) -> int | None:
    distinct = sorted(set(nums), reverse=True)
    return distinct[1] if len(distinct) >= 2 else None
""",
    buggy_solutions=(
        # Obvious: does not remove duplicates, so [7, 7, 7] returns 7.
        """\
def second_largest(nums: list[int]) -> int | None:
    return sorted(nums)[-2]
""",
        # Subtle: removes the maximum everywhere, then takes max of the rest. Crashes on [].
        """\
def second_largest(nums: list[int]) -> int | None:
    top = max(nums)
    rest = [x for x in nums if x != top]
    return max(rest) if rest else None
""",
    ),
    difficulty=1.0,
    split="train",
)

_FLATTEN = Task(
    id="flatten_nested",
    title="Flatten nested lists",
    prompt=(
        "Flatten arbitrarily nested lists into one flat list of the non-list values, in their "
        "original order. Only list instances are flattened; every other value, including "
        "0, None, False and the empty string, is kept as it is. Empty nested lists contribute "
        "nothing."
    ),
    signature="def flatten(items: list) -> list:",
    visible_tests=(
        "assert flatten([1, [2, [3]], 4]) == [1, 2, 3, 4]",
        "assert flatten([]) == []",
        "assert flatten([[], [[]]]) == []",
    ),
    hidden_tests=(
        "assert flatten([[1, 2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]",
        "assert flatten(['ab', ['c']]) == ['ab', 'c']",
        "assert flatten([0, [None, False]]) == [0, None, False]",
        "assert flatten([[[[]]], 7]) == [7]",
        "assert flatten([1, [1, [1]]]) == [1, 1, 1]",
    ),
    reference_solution="""\
def flatten(items: list) -> list:
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
""",
    buggy_solutions=(
        # Obvious: flattens only one level.
        """\
def flatten(items: list) -> list:
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out
""",
        # Subtle: skips falsy items, which drops 0, None and False.
        """\
def flatten(items: list) -> list:
    out = []
    for item in items:
        if not item:
            continue
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
""",
    ),
    difficulty=0.8,
    split="holdout",
)

_CHUNK = Task(
    id="chunk_list",
    title="Split list into chunks",
    prompt=(
        "Split items into consecutive sublists of length size. Every chunk has size elements "
        "except possibly the last, which holds the remainder. An empty input returns []. size "
        "is a positive integer. Do not modify the input list."
    ),
    signature="def chunk(items: list, size: int) -> list[list]:",
    visible_tests=(
        "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
        "assert chunk([1, 2, 3], 3) == [[1, 2, 3]]",
        "assert chunk([], 4) == []",
    ),
    hidden_tests=(
        "assert chunk([1], 1) == [[1]]",
        "assert chunk([1, 2], 5) == [[1, 2]]",
        "assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
        "assert chunk(['a', 'b', 'c'], 1) == [['a'], ['b'], ['c']]",
        "assert (lambda d: (chunk(d, 2), d)[1])([1, 2, 3]) == [1, 2, 3]",
    ),
    reference_solution="""\
def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]
""",
    buggy_solutions=(
        # Obvious: drops the final partial chunk.
        """\
def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items) - size + 1, size)]
""",
        # Subtle: empties the caller's list while chunking.
        """\
def chunk(items: list, size: int) -> list[list]:
    out = []
    while items:
        out.append(items[:size])
        del items[:size]
    return out
""",
    ),
    difficulty=0.7,
    split="holdout",
)

_LONGEST_PREFIX = Task(
    id="longest_common_prefix",
    title="Longest common prefix",
    prompt=(
        "Return the longest string that is a prefix of every string in words. Return '' if "
        "there is no common prefix or if words is empty. A string is a prefix of itself, so a "
        "single word is its own prefix. Comparison is case-sensitive."
    ),
    signature="def longest_common_prefix(words: list[str]) -> str:",
    visible_tests=(
        "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'",
        "assert longest_common_prefix(['dog', 'racecar', 'car']) == ''",
        "assert longest_common_prefix(['interview']) == 'interview'",
    ),
    hidden_tests=(
        "assert longest_common_prefix([]) == ''",
        "assert longest_common_prefix(['ab', 'abc', 'abcd']) == 'ab'",
        "assert longest_common_prefix(['a', 'a']) == 'a'",
        "assert longest_common_prefix(['abc', 'ab']) == 'ab'",
        "assert longest_common_prefix(['', 'abc']) == ''",
        "assert longest_common_prefix(['ab', 'x', 'ab']) == ''",
    ),
    reference_solution="""\
def longest_common_prefix(words: list[str]) -> str:
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
    return prefix
""",
    buggy_solutions=(
        # Obvious: returns the lexicographic minimum, which is not a common prefix.
        """\
def longest_common_prefix(words: list[str]) -> str:
    return min(words)
""",
        # Subtle: compares only the first and last words, which is right only when the
        # list is sorted.
        """\
def longest_common_prefix(words: list[str]) -> str:
    if not words:
        return ""
    first, last = words[0], words[-1]
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    return first[:i]
""",
    ),
    difficulty=1.1,
    split="holdout",
)

_IS_IPV4 = Task(
    id="valid_ipv4",
    title="Valid IPv4 address",
    prompt=(
        "Return True if s is a valid dotted-quad IPv4 address: exactly four parts separated by "
        "single dots, each part made only of ASCII digits, with a value from 0 to 255, and no "
        "leading zeros (so '0' is allowed but '01' is not). Spaces and any other characters "
        "make it invalid."
    ),
    signature="def is_valid_ipv4(s: str) -> bool:",
    visible_tests=(
        "assert is_valid_ipv4('192.168.0.1') is True",
        "assert is_valid_ipv4('256.1.1.1') is False",
        "assert is_valid_ipv4('1.2.3') is False",
    ),
    hidden_tests=(
        "assert is_valid_ipv4('0.0.0.0') is True",
        "assert is_valid_ipv4('01.2.3.4') is False",
        "assert is_valid_ipv4('1.2.3.4.5') is False",
        "assert is_valid_ipv4('1..2.3') is False",
        "assert is_valid_ipv4('1.2.3.') is False",
        "assert is_valid_ipv4('') is False",
    ),
    reference_solution="""\
def is_valid_ipv4(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not (p.isascii() and p.isdigit()):
            return False
        if len(p) > 1 and p[0] == "0":
            return False
        if int(p) > 255:
            return False
    return True
""",
    buggy_solutions=(
        # Obvious: counts dots only, so out-of-range values pass.
        """\
def is_valid_ipv4(s: str) -> bool:
    return s.count(".") == 3
""",
        # Subtle: accepts leading zeros such as '01'.
        """\
def is_valid_ipv4(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not (p.isascii() and p.isdigit()) or int(p) > 255:
            return False
    return True
""",
    ),
    difficulty=1.6,
    split="holdout",
)

_CORE_TASKS: tuple[Task, ...] = (
    _FIZZBUZZ,
    _REVERSE_WORDS,
    _IS_PALINDROME,
    _RLE_ENCODE,
    _ROMAN_TO_INT,
    _MERGE_INTERVALS,
    _BALANCED_BRACKETS,
    _SECOND_LARGEST,
    _FLATTEN,
    _CHUNK,
    _LONGEST_PREFIX,
    _IS_IPV4,
)


def _ordered(*groups: tuple[Task, ...]) -> tuple[Task, ...]:
    """Every train task, then every holdout task, each in group order."""
    tasks = [t for group in groups for t in group]
    return tuple(t for t in tasks if t.split == "train") + tuple(
        t for t in tasks if t.split == "holdout"
    )


TASKS: tuple[Task, ...] = _ordered(_CORE_TASKS, TEXT_TASKS, NUMBER_TASKS, DATA_TASKS)

_BY_ID: dict[str, Task] = {t.id: t for t in TASKS}


def load_benchmark(split: str = "train") -> list[Task]:
    """Return the tasks of a split ("train", "holdout" or "all") in a stable order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "all":
        return list(TASKS)
    return [t for t in TASKS if t.split == split]


def get_task(task_id: str) -> Task:
    """Return the task with the given id. Raises KeyError for an unknown id."""
    return _BY_ID[task_id]


def task_index() -> dict[str, Task]:
    """Return a fresh mapping from task id to task."""
    return dict(_BY_ID)
