"""Text-processing benchmark tasks: eight small string problems with visible and hidden tests.

Same contract as the coding benchmark in ``tasks.py``: every task carries a reference solution,
an obvious buggy solution (fails a visible test) and a subtle buggy solution (passes every
visible test but fails a hidden one).
"""

from evorg.bench.model import Task

_CAESAR_SHIFT = Task(
    id="caesar_shift",
    title="Caesar cipher",
    prompt=(
        "Apply a Caesar cipher to text: shift every ASCII letter forward by shift positions in "
        "the alphabet, wrapping from z back to a and from Z back to A. Preserve the case of each "
        "letter and leave every other character unchanged. The shift may be negative or larger "
        "than 26; a negative shift moves letters backwards. The empty string returns ''."
    ),
    signature="def caesar_encode(text: str, shift: int) -> str:",
    visible_tests=(
        "assert caesar_encode('abc', 1) == 'bcd'",
        "assert caesar_encode('xyz', 3) == 'abc'",
        "assert caesar_encode('Hello, World!', 13) == 'Uryyb, Jbeyq!'",
        "assert caesar_encode('', 5) == ''",
    ),
    hidden_tests=(
        "assert caesar_encode('abc', -1) == 'zab'",
        "assert caesar_encode('abc', 52) == 'abc'",
        "assert caesar_encode('Abc-Xyz', -30) == 'Wxy-Tuv'",
        "assert caesar_encode('ZZ', 1) == 'AA'",
        "assert caesar_encode('a1 b', 25) == 'z1 a'",
    ),
    reference_solution="""\
def caesar_encode(text: str, shift: int) -> str:
    k = shift % 26
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)
""",
    buggy_solutions=(
        # Obvious: shifts code points without wrapping, so 'xyz' by 3 becomes '{|}'.
        """\
def caesar_encode(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch.isascii() and ch.isalpha():
            out.append(chr(ord(ch) + shift))
        else:
            out.append(ch)
    return "".join(out)
""",
        # Subtle: negative shifts are treated as positive ones, so -1 behaves like +1.
        """\
def caesar_encode(text: str, shift: int) -> str:
    k = shift % 26 if shift >= 0 else (-shift) % 26
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)
""",
    ),
    difficulty=0.3,
    split="train",
)

_TOP_K_WORDS = Task(
    id="top_k_words",
    title="Top k frequent words",
    prompt=(
        "Return the k most frequent words in text, most frequent first. A word is a maximal run "
        "of ASCII letters; counting ignores case and the returned words are lowercase. Words "
        "with equal counts are ordered alphabetically. Return at most k words, and [] when k is "
        "zero or negative or when text contains no words."
    ),
    signature="def top_k_words(text: str, k: int) -> list[str]:",
    visible_tests=(
        "assert top_k_words('the cat and the hat and the bat', 2) == ['the', 'and']",
        "assert top_k_words('b a b a c', 3) == ['a', 'b', 'c']",
        "assert top_k_words('', 3) == []",
        "assert top_k_words('Go go GO stop', 1) == ['go']",
    ),
    hidden_tests=(
        "assert top_k_words('one two', 0) == []",
        "assert top_k_words('x y z', 10) == ['x', 'y', 'z']",
        "assert top_k_words(\"it's a dog's life\", 2) == ['s', 'a']",
        "assert top_k_words('Z z y', 2) == ['z', 'y']",
        "assert top_k_words('a1b2 c', 5) == ['a', 'b', 'c']",
    ),
    reference_solution="""\
import re


def top_k_words(text: str, k: int) -> list[str]:
    counts: dict[str, int] = {}
    for word in re.findall(r"[a-z]+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda w: (-counts[w], w))
    return ranked[:max(k, 0)]
""",
    buggy_solutions=(
        # Obvious: returns words in order of first appearance and never ranks them.
        """\
import re


def top_k_words(text: str, k: int) -> list[str]:
    seen = []
    for word in re.findall(r"[a-z]+", text.lower()):
        if word not in seen:
            seen.append(word)
    return seen[:max(k, 0)]
""",
        # Subtle: a non-positive k is treated as 1 because of "k or 1".
        """\
import re


def top_k_words(text: str, k: int) -> list[str]:
    counts: dict[str, int] = {}
    for word in re.findall(r"[a-z]+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda w: (-counts[w], w))
    return ranked[: k or 1]
""",
    ),
    difficulty=0.5,
    split="train",
)

_RLE_DECODE = Task(
    id="rle_decode",
    title="Run-length decoding",
    prompt=(
        "Decode a run-length encoded string in the format produced by run-length encoding: each "
        "run is a decimal count followed by one character that is repeated that many times, so "
        "'3a1b2c' becomes 'aaabcc'. Counts may have several digits, so '12a' becomes twelve 'a' "
        "characters. The run characters are never digits. The empty string decodes to ''."
    ),
    signature="def rle_decode(s: str) -> str:",
    visible_tests=(
        "assert rle_decode('3a1b2c') == 'aaabcc'",
        "assert rle_decode('') == ''",
        "assert rle_decode('4x') == 'xxxx'",
        "assert rle_decode('1a1b1c') == 'abc'",
    ),
    hidden_tests=(
        "assert rle_decode('12a') == 'a' * 12",
        "assert rle_decode('10x') == 'x' * 10",
        "assert rle_decode('2a11b') == 'aa' + 'b' * 11",
        "assert rle_decode('2 1-') == '  -'",
        "assert rle_decode('20Z') == 'Z' * 20",
    ),
    reference_solution="""\
def rle_decode(s: str) -> str:
    out = []
    count = ""
    for ch in s:
        if ch in "0123456789":
            count += ch
        else:
            out.append(ch * int(count))
            count = ""
    return "".join(out)
""",
    buggy_solutions=(
        # Obvious: keeps only the letters and drops the counts, so '3a1b2c' becomes 'abc'.
        """\
def rle_decode(s: str) -> str:
    return "".join(c for c in s if c.isalpha())
""",
        # Subtle: reads each count as a single digit, so multi-digit counts break.
        """\
def rle_decode(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        n = int(s[i])
        out.append(s[i + 1] * n)
        i += 2
    return "".join(out)
""",
    ),
    difficulty=0.7,
    split="train",
)

_GROUP_ANAGRAMS = Task(
    id="group_anagrams",
    title="Group anagrams",
    prompt=(
        "Group the words that are anagrams of each other. Two words are anagrams when they "
        "contain the same letters with the same counts; letter case matters. Within each group, "
        "sort the words alphabetically and keep duplicates. Return the groups ordered by their "
        "first word. Return [] for an empty list."
    ),
    signature="def group_anagrams(words: list[str]) -> list[list[str]]:",
    visible_tests=(
        "assert group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == "
        "[['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]",
        "assert group_anagrams([]) == []",
        "assert group_anagrams(['abc']) == [['abc']]",
    ),
    hidden_tests=(
        "assert group_anagrams(['Tea', 'eat']) == [['Tea'], ['eat']]",
        "assert group_anagrams(['ab', 'ab']) == [['ab', 'ab']]",
        "assert group_anagrams(['ab', 'ba', 'a']) == [['a'], ['ab', 'ba']]",
        "assert group_anagrams(['', '']) == [['', '']]",
    ),
    reference_solution="""\
def group_anagrams(words: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for word in words:
        groups.setdefault("".join(sorted(word)), []).append(word)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
""",
    buggy_solutions=(
        # Obvious: returns the groups in insertion order without sorting them.
        """\
def group_anagrams(words: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for word in words:
        groups.setdefault("".join(sorted(word)), []).append(word)
    return list(groups.values())
""",
        # Subtle: lowercases before building the key, so 'Tea' and 'eat' land together.
        """\
def group_anagrams(words: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for word in words:
        groups.setdefault("".join(sorted(word.lower())), []).append(word)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
""",
    ),
    difficulty=0.8,
    split="train",
)

_TO_SNAKE_CASE = Task(
    id="to_snake_case",
    title="camelCase to snake_case",
    prompt=(
        "Convert an identifier written in camelCase or PascalCase to snake_case. Insert an "
        "underscore before an uppercase letter that follows a lowercase letter or a digit, and "
        "before the last capital of an acronym run that is followed by a lowercase letter, so "
        "'HTTPServer' becomes 'http_server'. Lowercase the result. Names already in snake_case "
        "are unchanged, and the empty string returns ''."
    ),
    signature="def to_snake_case(name: str) -> str:",
    visible_tests=(
        "assert to_snake_case('camelCaseName') == 'camel_case_name'",
        "assert to_snake_case('PascalCase') == 'pascal_case'",
        "assert to_snake_case('already_snake') == 'already_snake'",
        "assert to_snake_case('') == ''",
    ),
    hidden_tests=(
        "assert to_snake_case('HTTPServer') == 'http_server'",
        "assert to_snake_case('getHTTPResponse') == 'get_http_response'",
        "assert to_snake_case('version2Beta') == 'version2_beta'",
        "assert to_snake_case('parseURL2Text') == 'parse_url2_text'",
        "assert to_snake_case('ABC') == 'abc'",
    ),
    reference_solution="""\
import re


def to_snake_case(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1_\\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s)
    return s.lower()
""",
    buggy_solutions=(
        # Obvious: puts an underscore before every capital, so 'PascalCase' gets a leading one.
        """\
import re


def to_snake_case(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\\1", name).lower()
""",
        # Subtle: a digit before a capital does not start a new word, so 'version2Beta' fails.
        """\
import re


def to_snake_case(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1_\\2", name)
    s = re.sub(r"([a-z])([A-Z])", r"\\1_\\2", s)
    return s.lower()
""",
    ),
    difficulty=1.0,
    split="train",
)

_WRAP_WORDS = Task(
    id="wrap_words",
    title="Word wrap",
    prompt=(
        "Word-wrap text to lines of at most width characters. Split text on whitespace and pack "
        "the words greedily into lines, joining the words on a line with single spaces. A word "
        "longer than width gets a line of its own and is not split. Return the list of lines; "
        "an empty or all-whitespace text returns []. A line whose length equals width is allowed."
    ),
    signature="def wrap_words(text: str, width: int) -> list[str]:",
    visible_tests=(
        "assert wrap_words('the quick brown fox', 10) == ['the quick', 'brown fox']",
        "assert wrap_words('a b c', 6) == ['a b c']",
        "assert wrap_words('', 10) == []",
        "assert wrap_words('supercalifragilistic is long', 8) == "
        "['supercalifragilistic', 'is long']",
    ),
    hidden_tests=(
        "assert wrap_words('aaa bbb', 7) == ['aaa bbb']",
        "assert wrap_words('one  two\\nthree', 7) == ['one two', 'three']",
        "assert wrap_words('x', 1) == ['x']",
        "assert wrap_words('a bb ccc', 4) == ['a bb', 'ccc']",
        "assert wrap_words('   ', 5) == []",
    ),
    reference_solution="""\
def wrap_words(text: str, width: int) -> list[str]:
    lines = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
""",
    buggy_solutions=(
        # Obvious: never wraps, so the whole text comes back as one line.
        """\
def wrap_words(text: str, width: int) -> list[str]:
    return [text] if text.strip() else []
""",
        # Subtle: uses a strict limit, so a line that exactly fills width is broken early.
        """\
def wrap_words(text: str, width: int) -> list[str]:
    lines = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) < width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
""",
    ),
    difficulty=1.2,
    split="train",
)

_TITLE_CASE = Task(
    id="title_case",
    title="Title case with small words",
    prompt=(
        "Convert a phrase to title case. Words are separated by single spaces. Capitalize the "
        "first letter of each word and lowercase the rest. Small words (a, an, and, as, at, but, "
        "by, for, in, of, on, or, the, to) stay lowercase, except when they are the first word, "
        "which is always capitalized. The empty string returns ''."
    ),
    signature="def title_case(s: str) -> str:",
    visible_tests=(
        "assert title_case('the lord of the rings') == 'The Lord of the Rings'",
        "assert title_case('a tale of two cities') == 'A Tale of Two Cities'",
        "assert title_case('') == ''",
    ),
    hidden_tests=(
        "assert title_case('of mice and men') == 'Of Mice and Men'",
        "assert title_case('HELLO wORLD') == 'Hello World'",
        "assert title_case('war and peace') == 'War and Peace'",
        "assert title_case('x') == 'X'",
        "assert title_case('in the end') == 'In the End'",
        "assert title_case('and') == 'And'",
    ),
    reference_solution="""\
def title_case(s: str) -> str:
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to"}
    out = []
    for i, word in enumerate(s.split(" ")):
        lower = word.lower()
        if i > 0 and lower in small:
            out.append(lower)
        else:
            out.append(lower[:1].upper() + lower[1:])
    return " ".join(out)
""",
    buggy_solutions=(
        # Obvious: capitalizes every word, so 'the lord of the rings' gets 'Of' and 'The'.
        """\
def title_case(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))
""",
        # Subtle: leaves the rest of each word as typed, so 'wORLD' becomes 'WORLD'.
        """\
def title_case(s: str) -> str:
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to"}
    out = []
    for i, word in enumerate(s.split(" ")):
        lower = word.lower()
        if i > 0 and lower in small:
            out.append(lower)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)
""",
    ),
    difficulty=1.3,
    split="holdout",
)

_LEVENSHTEIN = Task(
    id="levenshtein_distance",
    title="Levenshtein edit distance",
    prompt=(
        "Return the Levenshtein edit distance between strings a and b: the minimum number of "
        "single-character insertions, deletions and substitutions that turn a into b. "
        "Comparison is case-sensitive. The distance from an empty string to another string is "
        "the length of that string."
    ),
    signature="def levenshtein(a: str, b: str) -> int:",
    visible_tests=(
        "assert levenshtein('kitten', 'sitting') == 3",
        "assert levenshtein('', 'abc') == 3",
        "assert levenshtein('flaw', 'lawn') == 2",
    ),
    hidden_tests=(
        "assert levenshtein('', '') == 0",
        "assert levenshtein('abc', 'abc') == 0",
        "assert levenshtein('Flaw', 'flaw') == 1",
        "assert levenshtein('intention', 'execution') == 5",
        "assert levenshtein('sunday', 'saturday') == 3",
        "assert levenshtein('a', '') == 1",
    ),
    reference_solution="""\
def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
""",
    buggy_solutions=(
        # Obvious: compares lengths only, so 'kitten' and 'sitting' give 1 instead of 3.
        """\
def levenshtein(a: str, b: str) -> int:
    return abs(len(a) - len(b))
""",
        # Subtle: a substitution is free when the letters differ only in case.
        """\
def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = ca.lower() != cb.lower()
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]
""",
    ),
    difficulty=1.8,
    split="holdout",
)

TEXT_TASKS: tuple[Task, ...] = (
    _CAESAR_SHIFT,
    _TOP_K_WORDS,
    _RLE_DECODE,
    _GROUP_ANAGRAMS,
    _TO_SNAKE_CASE,
    _WRAP_WORDS,
    _TITLE_CASE,
    _LEVENSHTEIN,
)
