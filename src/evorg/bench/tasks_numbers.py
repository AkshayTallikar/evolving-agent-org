"""Numeric benchmark tasks: digits, divisibility, conversions and small sequences.

Same shape as ``tasks.py``: every task carries a reference solution, an obvious buggy
solution (fails a visible test) and a subtle buggy solution (passes every visible test but
fails a hidden one).
"""

from __future__ import annotations

from evorg.bench.model import Task

_DIGITAL_ROOT = Task(
    id="digital_root",
    title="Digital root",
    prompt=(
        "Return the digital root of the non-negative integer n: repeatedly replace the number "
        "by the sum of its decimal digits until a single digit remains. The digital root of 0 "
        "is 0, and the digital root of any positive number is between 1 and 9 inclusive."
    ),
    signature="def digital_root(n: int) -> int:",
    visible_tests=(
        "assert digital_root(0) == 0",
        "assert digital_root(38) == 2",
        "assert digital_root(493193) == 2",
    ),
    hidden_tests=(
        "assert digital_root(9) == 9",
        "assert digital_root(18) == 9",
        "assert digital_root(1) == 1",
        "assert digital_root(99) == 9",
        "assert digital_root(10) == 1",
    ),
    reference_solution="""\
def digital_root(n: int) -> int:
    if n == 0:
        return 0
    return 1 + (n - 1) % 9
""",
    buggy_solutions=(
        # Obvious: sums the digits once and never repeats, so 38 becomes 11.
        """\
def digital_root(n: int) -> int:
    return sum(int(c) for c in str(n))
""",
        # Subtle: n % 9 is right for everything except multiples of 9, which become 0, not 9.
        """\
def digital_root(n: int) -> int:
    return n % 9
""",
    ),
    difficulty=0.3,
    split="train",
)

_GCD_LIST = Task(
    id="gcd_list",
    title="GCD of a list",
    prompt=(
        "Return the greatest common divisor of all integers in nums, as a non-negative int. "
        "Negative values count by their absolute value, and zeros do not change the result "
        "(gcd(0, x) is |x|). If every value is zero, or nums is empty, return 0."
    ),
    signature="def gcd_list(nums: list[int]) -> int:",
    visible_tests=(
        "assert gcd_list([12, 18, 24]) == 6",
        "assert gcd_list([]) == 0",
        "assert gcd_list([7]) == 7",
    ),
    hidden_tests=(
        "assert gcd_list([0, 0]) == 0",
        "assert gcd_list([0, 9]) == 9",
        "assert gcd_list([-12, -18]) == 6",
        "assert gcd_list([6, -4]) == 2",
        "assert gcd_list([-12]) == 12",
    ),
    reference_solution="""\
def gcd_list(nums: list[int]) -> int:
    g = 0
    for x in nums:
        a, b = abs(g), abs(x)
        while b:
            a, b = b, a % b
        g = a
    return g
""",
    buggy_solutions=(
        # Obvious: returns the first value, which is only the gcd by coincidence.
        """\
def gcd_list(nums: list[int]) -> int:
    return nums[0] if nums else 0
""",
        # Subtle: plain Euclid without abs(). Python's % keeps the sign of the divisor, so a
        # negative second argument can leave a negative result. Positive inputs are fine.
        """\
def gcd_list(nums: list[int]) -> int:
    g = 0
    for x in nums:
        a, b = g, x
        while b:
            a, b = b, a % b
        g = a
    return g
""",
    ),
    difficulty=0.5,
    split="train",
)

_INT_TO_ROMAN = Task(
    id="int_to_roman",
    title="Integer to Roman numeral",
    prompt=(
        "Convert the integer n, with 1 <= n <= 3999, to a Roman numeral written in uppercase "
        "letters I, V, X, L, C, D and M. Use the subtractive forms IV, IX, XL, XC, CD and CM "
        "where they apply, so 1994 becomes 'MCMXCIV' and 444 becomes 'CDXLIV'."
    ),
    signature="def int_to_roman(n: int) -> str:",
    visible_tests=(
        "assert int_to_roman(3) == 'III'",
        "assert int_to_roman(58) == 'LVIII'",
        "assert int_to_roman(1994) == 'MCMXCIV'",
    ),
    hidden_tests=(
        "assert int_to_roman(1) == 'I'",
        "assert int_to_roman(9) == 'IX'",
        "assert int_to_roman(40) == 'XL'",
        "assert int_to_roman(444) == 'CDXLIV'",
        "assert int_to_roman(3999) == 'MMMCMXCIX'",
    ),
    reference_solution="""\
def int_to_roman(n: int) -> str:
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
        (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for value, symbol in table:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)
""",
    buggy_solutions=(
        # Obvious: additive symbols only, so 1994 becomes MDCCCCLXXXXIIII.
        """\
def int_to_roman(n: int) -> str:
    table = [(1000, "M"), (500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V"), (1, "I")]
    out = []
    for value, symbol in table:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)
""",
        # Subtle: the table lacks the XL (40) pair, so 40 becomes XXXX. The visible numerals
        # never need 40, so they all pass.
        """\
def int_to_roman(n: int) -> str:
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
        (50, "L"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for value, symbol in table:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)
""",
    ),
    difficulty=0.9,
    split="train",
)

_PRIME_FACTORS = Task(
    id="prime_factors",
    title="Prime factorisation",
    prompt=(
        "Return the prime factors of n with multiplicity, in ascending order: 12 gives "
        "[2, 2, 3]. If n is less than 2 return an empty list. Use trial division; n is an "
        "integer with at most about 10 digits."
    ),
    signature="def prime_factors(n: int) -> list[int]:",
    visible_tests=(
        "assert prime_factors(12) == [2, 2, 3]",
        "assert prime_factors(97) == [97]",
        "assert prime_factors(1) == []",
    ),
    hidden_tests=(
        "assert prime_factors(0) == []",
        "assert prime_factors(-5) == []",
        "assert prime_factors(2) == [2]",
        "assert prime_factors(4) == [2, 2]",
        "assert prime_factors(9) == [3, 3]",
        "assert prime_factors(360) == [2, 2, 2, 3, 3, 5]",
    ),
    reference_solution="""\
def prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
""",
    buggy_solutions=(
        # Obvious: divides out each prime completely, so it keeps distinct primes only and
        # 12 becomes [2, 3].
        """\
def prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
""",
        # Subtle: the loop bound is strict, so a perfect square of a prime such as 4 or 9
        # is never split. The visible numbers have no such square factor.
        """\
def prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d < n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
""",
    ),
    difficulty=1.0,
    split="train",
)

_MAX_SUBARRAY = Task(
    id="max_subarray_sum",
    title="Maximum subarray sum",
    prompt=(
        "Return the largest sum of a contiguous, non-empty run of elements in nums. The run "
        "may consist of a single element, so when every value is negative the answer is the "
        "largest single value. An empty list returns 0."
    ),
    signature="def max_subarray_sum(nums: list[int]) -> int:",
    visible_tests=(
        "assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6",
        "assert max_subarray_sum([1, 2, 3]) == 6",
        "assert max_subarray_sum([]) == 0",
    ),
    hidden_tests=(
        "assert max_subarray_sum([-3, -1, -2]) == -1",
        "assert max_subarray_sum([-7]) == -7",
        "assert max_subarray_sum([5, -9, 3]) == 5",
        "assert max_subarray_sum([4, -1, -2, 1, 5]) == 7",
        "assert max_subarray_sum([0, 0]) == 0",
    ),
    reference_solution="""\
def max_subarray_sum(nums: list[int]) -> int:
    if not nums:
        return 0
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
""",
    buggy_solutions=(
        # Obvious: sums the whole list, which is 1 for the first visible example, not 6.
        """\
def max_subarray_sum(nums: list[int]) -> int:
    return sum(nums)
""",
        # Subtle: starts best at 0, which is the empty subarray. Correct whenever some
        # subarray is non-negative, but returns 0 instead of -1 for [-3, -1, -2].
        """\
def max_subarray_sum(nums: list[int]) -> int:
    best = cur = 0
    for x in nums:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
""",
    ),
    difficulty=1.5,
    split="train",
)

_INT_TO_BASE = Task(
    id="int_to_base",
    title="Integer to string in a base",
    prompt=(
        "Return the integer n written in the given base, 2 <= base <= 36, as a string of "
        "uppercase digits 0-9 and A-Z, where A is 10 and Z is 35. Zero is '0'. A negative n "
        "gets a leading '-' and is written by its absolute value. There is no prefix such as "
        "'0x'."
    ),
    signature="def to_base(n: int, base: int) -> str:",
    visible_tests=(
        "assert to_base(10, 2) == '1010'",
        "assert to_base(255, 16) == 'FF'",
        "assert to_base(0, 7) == '0'",
    ),
    hidden_tests=(
        "assert to_base(-10, 2) == '-1010'",
        "assert to_base(1, 2) == '1'",
        "assert to_base(35, 36) == 'Z'",
        "assert to_base(12345, 36) == '9IX'",
    ),
    reference_solution="""\
def to_base(n: int, base: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    return sign + "".join(reversed(out))
""",
    buggy_solutions=(
        # Obvious: ignores the base and returns the decimal form, so 10 in base 2 is '10'.
        """\
def to_base(n: int, base: int) -> str:
    return str(n)
""",
        # Subtle: the digit table stops at F, so digits above 15 raise IndexError. Bases up
        # to 16 are fine, which is all the visible tests use.
        """\
def to_base(n: int, base: int) -> str:
    digits = "0123456789ABCDEF"
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    return sign + "".join(reversed(out))
""",
    ),
    difficulty=1.2,
    split="holdout",
)

_LEAP_YEARS = Task(
    id="leap_years_between",
    title="Leap years in a range",
    prompt=(
        "Return the Gregorian leap years y with start <= y <= end, in ascending order, as a "
        "list. A year is a leap year if it is divisible by 4, except century years, which "
        "are leap years only when divisible by 400. Both ends are included, and if start is "
        "greater than end the result is an empty list."
    ),
    signature="def leap_years_between(start: int, end: int) -> list[int]:",
    visible_tests=(
        "assert leap_years_between(2000, 2010) == [2000, 2004, 2008]",
        "assert leap_years_between(1900, 1900) == []",
        "assert leap_years_between(2024, 2023) == []",
    ),
    hidden_tests=(
        "assert leap_years_between(2024, 2024) == [2024]",
        "assert leap_years_between(1996, 2000) == [1996, 2000]",
        "assert leap_years_between(1, 4) == [4]",
        "assert leap_years_between(2100, 2100) == []",
        "assert leap_years_between(1600, 1604) == [1600, 1604]",
    ),
    reference_solution="""\
def leap_years_between(start: int, end: int) -> list[int]:
    return [
        y for y in range(start, end + 1)
        if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    ]
""",
    buggy_solutions=(
        # Obvious: ignores century years, so 1900 is wrongly reported as a leap year.
        """\
def leap_years_between(start: int, end: int) -> list[int]:
    return [y for y in range(start, end + 1) if y % 4 == 0]
""",
        # Subtle: the range excludes end, so a range whose last year is a leap year loses it.
        # The visible ranges end on a non-leap year or are empty.
        """\
def leap_years_between(start: int, end: int) -> list[int]:
    return [
        y for y in range(start, end)
        if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    ]
""",
    ),
    difficulty=0.7,
    split="holdout",
)

_PASCAL_ROW = Task(
    id="pascal_row",
    title="Row of Pascal's triangle",
    prompt=(
        "Return row n of Pascal's triangle as a list of ints, where row 0 is [1] and each "
        "later row starts and ends with 1 and holds the sums of adjacent pairs of the row "
        "above it. For n = 4 the result is [1, 4, 6, 4, 1]. If n is negative, return an "
        "empty list."
    ),
    signature="def pascal_row(n: int) -> list[int]:",
    visible_tests=(
        "assert pascal_row(0) == [1]",
        "assert pascal_row(3) == [1, 3, 3, 1]",
        "assert pascal_row(4) == [1, 4, 6, 4, 1]",
    ),
    hidden_tests=(
        "assert pascal_row(1) == [1, 1]",
        "assert pascal_row(2) == [1, 2, 1]",
        "assert pascal_row(-3) == []",
        "assert sum(pascal_row(10)) == 1024",
        "assert pascal_row(10)[5] == 252",
    ),
    reference_solution="""\
def pascal_row(n: int) -> list[int]:
    if n < 0:
        return []
    row = [1]
    for _ in range(n):
        row = [1] + [a + b for a, b in zip(row, row[1:])] + [1]
    return row
""",
    buggy_solutions=(
        # Obvious: fills every position with 1, so row 4 becomes [1, 1, 1, 1, 1].
        """\
def pascal_row(n: int) -> list[int]:
    return [1] * (n + 1) if n >= 0 else []
""",
        # Subtle: the negative-input guard is missing, so range(n) is empty and the result is
        # [1] instead of []. Non-negative rows are computed correctly.
        """\
def pascal_row(n: int) -> list[int]:
    row = [1]
    for _ in range(n):
        row = [1] + [a + b for a, b in zip(row, row[1:])] + [1]
    return row
""",
    ),
    difficulty=1.8,
    split="holdout",
)

NUMBER_TASKS: tuple[Task, ...] = (
    _DIGITAL_ROOT,
    _GCD_LIST,
    _INT_TO_ROMAN,
    _PRIME_FACTORS,
    _MAX_SUBARRAY,
    _INT_TO_BASE,
    _LEAP_YEARS,
    _PASCAL_ROW,
)
