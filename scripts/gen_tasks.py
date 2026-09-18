"""Generate verifiable terminal tasks.

Every task is a (setup, instruction, check) triple. Setup and check are bash. Ground truth
is computed in Python at generation time, so the checker is exact. Train and test splits
use disjoint seeds; every template appears in both splits.

    python scripts/gen_tasks.py --train 320 --test 80 --out tasks/
"""
from __future__ import annotations

import argparse
import os
import random
import shlex
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from termbench.tasks import Task, save_tasks  # noqa: E402

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu apple banana "
         "cherry grape lemon mango olive peach plum berry maple cedar pine oak birch river stone cloud").split()
EXTS = ["txt", "log", "csv", "md", "py", "json", "cfg", "dat"]
DIRS = ["data", "logs", "src", "docs", "build", "assets", "notes", "tmp", "config", "reports"]


def q(s: str) -> str:
    return shlex.quote(s)


def write_lines(path: str, lines: list[str]) -> str:
    """bash snippet writing lines to path (creating parent dirs)."""
    d = os.path.dirname(path)
    mk = f"mkdir -p {q(d)} && " if d else ""
    body = "".join(f"{q(line)} " for line in lines)
    return f"{mk}printf '%s\\n' {body}> {q(path)}"


def answer_check(expected: str) -> str:
    return f'[ "$(tr -d \'[:space:]\' < answer.txt 2>/dev/null)" = {q(expected.strip())} ]'


def rand_lines(rng: random.Random, n: int, k: int = 4) -> list[str]:
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, k))) for _ in range(n)]


def fname(rng: random.Random, ext: str | None = None) -> str:
    return f"{rng.choice(WORDS)}_{rng.randint(1, 99)}.{ext or rng.choice(EXTS)}"


# ---- templates: each returns (category, instruction, setup, check, difficulty) --------------

def t_count_lines(rng):
    d, f = rng.choice(DIRS), fname(rng, "txt")
    lines = rand_lines(rng, rng.randint(5, 60))
    p = f"{d}/{f}"
    return ("count_lines", f"Count the number of lines in {p} and write just the number to answer.txt.",
            write_lines(p, lines), answer_check(str(len(lines))), 1)


def t_count_words(rng):
    f = fname(rng, "txt")
    lines = rand_lines(rng, rng.randint(3, 20), 6)
    n = sum(len(l.split()) for l in lines)
    return ("count_words", f"How many words are in {f}? Write only the number to answer.txt.",
            write_lines(f, lines), answer_check(str(n)), 1)


def t_grep_count(rng):
    d, f = rng.choice(DIRS), fname(rng, "log")
    word = rng.choice(WORDS)
    lines = rand_lines(rng, rng.randint(10, 50))
    n = sum(1 for l in lines if word in l.split())
    return ("grep_count", f"Count how many lines in {d}/{f} contain the word '{word}' and write only that number to answer.txt.",
            write_lines(f"{d}/{f}", lines), answer_check(str(n)), 1)


def t_find_largest(rng):
    d = rng.choice(DIRS)
    files = {fname(rng, "bin") for _ in range(rng.randint(3, 6))}
    sizes = rng.sample(range(100, 5000, 37), len(files))
    setup = f"mkdir -p {q(d)} && " + " && ".join(f"head -c {s} /dev/zero > {q(d + '/' + f)}" for f, s in zip(files, sizes))
    biggest = max(zip(files, sizes), key=lambda x: x[1])[0]
    return ("find_largest", f"Find the largest file inside {d}/ and write its file name (not the full path) to answer.txt.",
            setup, answer_check(biggest), 2)


def t_rename_ext(rng):
    d = rng.choice(DIRS)
    names = list({rng.choice(WORDS) + str(rng.randint(1, 50)) for _ in range(rng.randint(2, 5))})
    other = fname(rng, "csv")
    setup = " && ".join([f"mkdir -p {q(d)}"] + [write_lines(f"{d}/{n}.txt", rand_lines(rng, 2)) for n in names]
                        + [write_lines(f"{d}/{other}", ["keep"])])
    check = " && ".join([f"[ -f {q(d + '/' + n + '.md')} ]" for n in names]
                        + [f"[ ! -f {q(d + '/' + n + '.txt')} ]" for n in names] + [f"[ -f {q(d + '/' + other)} ]"])
    return ("rename_ext", f"Rename every .txt file in {d}/ so it has the .md extension instead (keep other files untouched).",
            setup, check, 2)


def t_mkdir_tree(rng):
    parts = [rng.choice(WORDS) for _ in range(3)]
    path = "/".join(parts)
    f = fname(rng, "txt")
    return ("mkdir_tree", f"Create the directory path {path}/ and inside it an empty file named {f}.",
            "true", f"[ -d {q(path)} ] && [ -f {q(path + '/' + f)} ]", 1)


def t_copy_file(rng):
    src_d, dst_d = rng.sample(DIRS, 2)
    f = fname(rng, "cfg")
    lines = rand_lines(rng, 3)
    return ("copy_file", f"Copy {src_d}/{f} into the directory {dst_d}/ (create it if needed), keeping the original.",
            write_lines(f"{src_d}/{f}", lines),
            f"[ -f {q(src_d + '/' + f)} ] && cmp -s {q(src_d + '/' + f)} {q(dst_d + '/' + f)}", 1)


def t_move_file(rng):
    src_d, dst_d = rng.sample(DIRS, 2)
    f = fname(rng, "dat")
    lines = rand_lines(rng, 3)
    return ("move_file", f"Move {src_d}/{f} into {dst_d}/ (create the directory if needed).",
            write_lines(f"{src_d}/{f}", lines) + f" && mkdir -p {q(dst_d)}",
            f"[ ! -e {q(src_d + '/' + f)} ] && [ -f {q(dst_d + '/' + f)} ] && [ \"$(head -1 {q(dst_d + '/' + f)})\" = {q(lines[0])} ]", 1)


def t_delete_pattern(rng):
    d = rng.choice(DIRS)
    tmp = list({fname(rng, "tmp") for _ in range(rng.randint(2, 4))})
    keep = list({fname(rng, rng.choice(["txt", "log"])) for _ in range(rng.randint(2, 4))})
    setup = " && ".join([f"mkdir -p {q(d)}"] + [f"echo x > {q(d + '/' + f)}" for f in tmp + keep])
    check = " && ".join([f"[ ! -e {q(d + '/' + f)} ]" for f in tmp] + [f"[ -f {q(d + '/' + f)} ]" for f in keep])
    return ("delete_pattern", f"Delete all files ending in .tmp inside {d}/ and nothing else.", setup, check, 1)


def t_sort_unique(rng):
    f = fname(rng, "txt")
    pool = rng.sample(WORDS, rng.randint(4, 8))
    lines = [rng.choice(pool) for _ in range(rng.randint(10, 25))]
    return ("sort_unique", f"Write the sorted, de-duplicated lines of {f} into unique.txt (one per line).",
            write_lines(f, lines), f"sort -u {q(f)} | diff -q - unique.txt >/dev/null", 1)


def t_sum_column(rng):
    f = fname(rng, "csv")
    n = rng.randint(4, 12)
    rows = [f"{rng.choice(WORDS)},{rng.randint(1, 500)},{rng.choice(WORDS)}" for _ in range(n)]
    total = sum(int(r.split(",")[1]) for r in rows)
    return ("sum_column", f"{f} is a comma-separated file with no header. Sum the numbers in the second column and write only the total to answer.txt.",
            write_lines(f, rows), answer_check(str(total)), 2)


def t_replace_text(rng):
    d, f = rng.choice(DIRS), fname(rng, "md")
    old, new = rng.sample(WORDS, 2)
    lines = rand_lines(rng, rng.randint(4, 12))
    if not any(old in l.split() for l in lines):
        lines[rng.randrange(len(lines))] += " " + old
    expected = [" ".join(new if w == old else w for w in l.split()) for l in lines]
    body = "".join(f"{q(line)} " for line in expected)
    return ("replace_text", f"In {d}/{f}, replace every occurrence of the word '{old}' with '{new}' (edit the file in place).",
            write_lines(f"{d}/{f}", lines),
            f"printf '%s\\n' {body}| cmp -s - {q(d + '/' + f)}", 2)


def t_word_freq(rng):
    f = fname(rng, "txt")
    pool = rng.sample(WORDS, 5)
    top = pool[0]
    counts = {w: rng.randint(1, 5) for w in pool}
    counts[top] = max(counts.values()) + rng.randint(1, 3)
    words = [w for w, c in counts.items() for _ in range(c)]
    rng.shuffle(words)
    lines = [" ".join(words[i:i + 4]) for i in range(0, len(words), 4)]
    return ("word_freq", f"Find the most frequent word in {f} and write only that word to answer.txt.",
            write_lines(f, lines), answer_check(top), 2)


def t_tar_extract(rng):
    d = rng.choice(DIRS)
    inner = fname(rng, "txt")
    lines = rand_lines(rng, 3)
    setup = (f"mkdir -p .mk && " + write_lines(f".mk/{inner}", lines)
             + f" && tar -czf archive.tar.gz -C .mk {q(inner)} && rm -rf .mk && mkdir -p {q(d)}")
    return ("tar_extract", f"Extract archive.tar.gz into the directory {d}/.",
            setup, f"[ -f {q(d + '/' + inner)} ] && [ \"$(head -1 {q(d + '/' + inner)})\" = {q(lines[0])} ]", 1)


def t_tar_create(rng):
    d = rng.choice(DIRS)
    files = list({fname(rng, "txt") for _ in range(rng.randint(2, 4))})
    setup = " && ".join([f"mkdir -p {q(d)}"] + [write_lines(f"{d}/{f}", rand_lines(rng, 2)) for f in files])
    check = "[ -f backup.tar.gz ] && " + " && ".join(f"tar -tzf backup.tar.gz | grep -q {q(f)}" for f in files)
    return ("tar_create", f"Create a gzip-compressed tar archive named backup.tar.gz containing the {d}/ directory.", setup, check, 1)


def t_chmod_exec(rng):
    f = rng.choice(WORDS) + ".sh"
    return ("chmod_exec", f"Make the script {f} executable.",
            write_lines(f, ["#!/bin/bash", "echo hi"]), f"[ -x {q(f)} ]", 1)


def t_symlink(rng):
    d, f = rng.choice(DIRS), fname(rng, "cfg")
    link = rng.choice(WORDS) + ".link"
    return ("symlink", f"Create a symbolic link named {link} in the current directory that points to {d}/{f}.",
            write_lines(f"{d}/{f}", ["v=1"]), f"[ -L {q(link)} ] && [ \"$(readlink {q(link)})\" = {q(d + '/' + f)} ]", 1)


def t_tail_lines(rng):
    f = fname(rng, "log")
    k = rng.randint(2, 5)
    lines = rand_lines(rng, rng.randint(k + 3, 25))
    return ("tail_lines", f"Write the last {k} lines of {f} into tail.txt (same order).",
            write_lines(f, lines), f"tail -n {k} {q(f)} | diff -q - tail.txt >/dev/null", 1)


def t_head_lines(rng):
    f = fname(rng, "log")
    k = rng.randint(2, 5)
    lines = rand_lines(rng, rng.randint(k + 3, 25))
    return ("head_lines", f"Write the first {k} lines of {f} into head.txt.",
            write_lines(f, lines), f"head -n {k} {q(f)} | diff -q - head.txt >/dev/null", 1)


def t_count_files_ext(rng):
    ext = rng.choice(["py", "log", "json"])
    dirs = rng.sample(DIRS, 3)
    n = 0
    parts = []
    for d in dirs:
        sub = f"{d}/{rng.choice(WORDS)}" if rng.random() < 0.5 else d
        for _ in range(rng.randint(0, 3)):
            parts.append(f"mkdir -p {q(sub)} && echo x > {q(sub + '/' + fname(rng, ext))}")
        for _ in range(rng.randint(0, 2)):
            parts.append(f"mkdir -p {q(sub)} && echo x > {q(sub + '/' + fname(rng, 'txt'))}")
    setup = " && ".join(parts) or "true"
    # count actual (names may collide in the set of parts, so count from the generated commands)
    names = set()
    for p in parts:
        if p.endswith(f".{ext}'") or p.endswith(f".{ext}"):
            names.add(p.split(">")[-1].strip())
    n = len(names)
    return ("count_files_ext", f"Count how many files with the .{ext} extension exist anywhere under the current directory (recursively) and write only the number to answer.txt.",
            setup, answer_check(str(n)), 2)


def t_find_by_content(rng):
    d = rng.choice(DIRS)
    files = list({fname(rng, "txt") for _ in range(rng.randint(3, 6))})
    needle = "NEEDLE" + str(rng.randint(100, 999))
    target = rng.choice(files)
    parts = [f"mkdir -p {q(d)}"]
    for f in files:
        lines = rand_lines(rng, 4)
        if f == target:
            lines.insert(rng.randrange(len(lines)), f"key {needle} here")
        parts.append(write_lines(f"{d}/{f}", lines))
    return ("find_by_content", f"Exactly one file in {d}/ contains the string {needle}. Write that file's name (not path) to answer.txt.",
            " && ".join(parts), answer_check(target), 2)


def t_json_extract(rng):
    key = rng.choice(["port", "retries", "timeout", "workers", "limit"])
    val = rng.randint(1, 9999)
    other = {k: rng.randint(1, 99) for k in rng.sample(["debug", "level", "size", "depth"], 2)}
    obj = {**other, key: val, "name": rng.choice(WORDS)}
    import json as _json
    text = _json.dumps(obj, indent=2)
    return ("json_extract", f"config.json is a JSON object. Write the value of its \"{key}\" field (just the number) to answer.txt.",
            write_lines("config.json", text.split("\n")), answer_check(str(val)), 2)


def t_append_line(rng):
    f = fname(rng, "txt")
    lines = rand_lines(rng, rng.randint(2, 6))
    new = " ".join(rng.sample(WORDS, 2))
    return ("append_line", f"Append the line \"{new}\" to the end of {f}.",
            write_lines(f, lines),
            f"[ \"$(tail -n 1 {q(f)})\" = {q(new)} ] && [ \"$(wc -l < {q(f)})\" -eq {len(lines) + 1} ]", 1)


def t_concat(rng):
    d = rng.choice(DIRS)
    names = ["part1.txt", "part2.txt", "part3.txt"]
    parts = [f"mkdir -p {q(d)}"] + [write_lines(f"{d}/{n}", rand_lines(rng, 2)) for n in names]
    return ("concat", f"Concatenate {d}/part1.txt, {d}/part2.txt and {d}/part3.txt in that order into combined.txt in the current directory.",
            " && ".join(parts), f"cat {q(d + '/part1.txt')} {q(d + '/part2.txt')} {q(d + '/part3.txt')} | diff -q - combined.txt >/dev/null", 1)


def t_max_number(rng):
    f = fname(rng, "dat")
    nums = rng.sample(range(1, 10000), rng.randint(5, 30))
    return ("max_number", f"{f} contains one integer per line. Write the largest one to answer.txt.",
            write_lines(f, [str(n) for n in nums]), answer_check(str(max(nums))), 1)


def t_cut_field(rng):
    f = rng.choice(["users.txt", "records.txt", "entries.txt"])
    n = rng.randint(3, 8)
    rows = [f"{rng.randint(1000, 9999)}:{rng.choice(WORDS)}:{rng.choice(WORDS)}@example.com" for _ in range(n)]
    return ("cut_field", f"{f} has colon-separated fields. Write the second field of every line, in order, to names.txt.",
            write_lines(f, rows), f"cut -d: -f2 {q(f)} | diff -q - names.txt >/dev/null", 1)


def t_gzip_file(rng):
    d, f = rng.choice(DIRS), fname(rng, "log")
    return ("gzip_file", f"Compress {d}/{f} with gzip so that {d}/{f}.gz exists and the uncompressed file is gone.",
            write_lines(f"{d}/{f}", rand_lines(rng, 8)), f"[ -f {q(d + '/' + f + '.gz')} ] && [ ! -e {q(d + '/' + f)} ] && gzip -t {q(d + '/' + f + '.gz')}", 1)


def t_empty_files(rng):
    d = rng.choice(DIRS)
    empty = list({fname(rng) for _ in range(rng.randint(2, 4))})
    full = list({fname(rng) for _ in range(rng.randint(2, 4))} - set(empty))
    setup = " && ".join([f"mkdir -p {q(d)}"] + [f": > {q(d + '/' + f)}" for f in empty] + [f"echo data > {q(d + '/' + f)}" for f in full])
    check = " && ".join([f"[ ! -e {q(d + '/' + f)} ]" for f in empty] + [f"[ -f {q(d + '/' + f)} ]" for f in full])
    return ("empty_files", f"Delete every empty (zero-byte) file in {d}/ and keep the rest.", setup, check, 2)


def t_reverse_lines(rng):
    f = fname(rng, "txt")
    lines = rand_lines(rng, rng.randint(4, 15))
    return ("reverse_lines", f"Write the lines of {f} in reverse order into reversed.txt.",
            write_lines(f, lines), f"tac {q(f)} | diff -q - reversed.txt >/dev/null", 1)


def t_char_count(rng):
    f = fname(rng, "txt")
    lines = rand_lines(rng, rng.randint(2, 8))
    n = sum(len(l) + 1 for l in lines)
    return ("char_count", f"How many bytes is {f}? Write only the number to answer.txt.",
            write_lines(f, lines), answer_check(str(n)), 1)


def t_dir_sizes(rng):
    dirs = rng.sample(DIRS, 3)
    sizes = rng.sample(range(2000, 60000, 1234), 3)
    setup = " && ".join(f"mkdir -p {q(d)} && head -c {s} /dev/zero > {q(d + '/blob.bin')}" for d, s in zip(dirs, sizes))
    biggest = max(zip(dirs, sizes), key=lambda x: x[1])[0]
    return ("dir_sizes", "Which top-level directory here uses the most disk space? Write only its name to answer.txt.",
            setup, answer_check(biggest), 2)


def t_line_number_match(rng):
    f = fname(rng, "log")
    lines = rand_lines(rng, rng.randint(8, 30))
    marker = "ERROR"
    i = rng.randrange(len(lines))
    lines[i] = f"{marker} " + lines[i]
    return ("line_number_match", f"On which line number (1-based) does the word {marker} first appear in {f}? Write only the number to answer.txt.",
            write_lines(f, lines), answer_check(str(i + 1)), 2)


def t_touch_many(rng):
    d = rng.choice(DIRS)
    n = rng.randint(3, 7)
    return ("touch_many", f"Inside a new directory {d}/, create empty files file1.txt through file{n}.txt.",
            "true", " && ".join(f"[ -f {q(d + '/file' + str(i) + '.txt')} ]" for i in range(1, n + 1)) + f" && [ \"$(ls {q(d)} | wc -l)\" -eq {n} ]", 1)


TEMPLATES = [t_count_lines, t_count_words, t_grep_count, t_find_largest, t_rename_ext, t_mkdir_tree,
             t_copy_file, t_move_file, t_delete_pattern, t_sort_unique, t_sum_column, t_replace_text,
             t_word_freq, t_tar_extract, t_tar_create, t_chmod_exec, t_symlink, t_tail_lines, t_head_lines,
             t_count_files_ext, t_find_by_content, t_json_extract, t_append_line, t_concat, t_max_number,
             t_cut_field, t_gzip_file, t_empty_files, t_reverse_lines, t_char_count, t_dir_sizes,
             t_line_number_match, t_touch_many]


def make_split(n: int, seed: int, prefix: str) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        cat, instr, setup, check, diff = tmpl(rng)
        tasks.append(Task(id=f"{prefix}_{i:04d}_{cat}", category=cat, instruction=instr, setup=setup,
                          check=check, difficulty=diff, max_turns=8))
    rng.shuffle(tasks)
    return tasks


def make_hard_split(n: int, seed: int, prefix: str) -> list[Task]:
    """Composite tasks: two independent sub-tasks from different templates in one instruction.
    Needs planning across steps and is where the SFT model (trained on single-step tasks)
    starts to fail, which is what RL needs: mixed outcomes within a group."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    tries = 0
    while len(tasks) < n and tries < n * 50:
        tries += 1
        ta, tb = rng.sample(TEMPLATES, 2)
        ca, ia, sa, cka, _ = ta(rng)
        cb, ib, sb, ckb, _ = tb(rng)
        if ca == cb or ("answer.txt" in cka and "answer.txt" in ckb):
            continue
        # crude fixture-collision guard: the two setups must not mention the same file names
        toks_a = {t for t in sa.replace("'", " ").split() if "." in t and "/" in t}
        toks_b = {t for t in sb.replace("'", " ").split() if "." in t and "/" in t}
        if toks_a & toks_b:
            continue
        instr = f"Do both of the following. (1) {ia} (2) {ib}"
        tasks.append(Task(id=f"{prefix}_{len(tasks):04d}_{ca}+{cb}", category="composite", instruction=instr,
                          setup=f"{sa} && {sb}", check=f"( {cka} ) && ( {ckb} )", difficulty=3, max_turns=8,
                          meta={"parts": [ca, cb]}))
    return tasks


def validate(tasks: list[Task]) -> None:
    """Run every setup + check with a reference solution-free sanity: setup must succeed and
    check must FAIL on the untouched fixture (otherwise the task is trivially solved)."""
    from termbench.sandbox import Sandbox

    bad = []
    for t in tasks:
        with Sandbox() as sb:
            r = sb.run(t.setup, timeout=30, check_policy=False)
            if r.exit_code != 0:
                bad.append((t.id, "setup failed: " + r.stderr[:200]))
                continue
            c = sb.run(t.check, timeout=30, check_policy=False)
            if c.exit_code == 0:
                bad.append((t.id, "check passes on untouched fixture"))
    if bad:
        for b in bad:
            print("BAD", b)
        raise SystemExit(f"{len(bad)} invalid tasks")
    print(f"validated {len(tasks)} tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=320)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--out", default="tasks")
    ap.add_argument("--validate", action="store_true", help="run setups/checks in a sandbox (Linux only)")
    ap.add_argument("--hard", action="store_true", help="only (re)generate the composite hard splits")
    ap.add_argument("--hard-train", type=int, default=160)
    ap.add_argument("--hard-test", type=int, default=48)
    a = ap.parse_args()
    if a.hard:
        htrain = make_hard_split(a.hard_train, seed=555, prefix="htrain")
        htest = make_hard_split(a.hard_test, seed=777, prefix="htest")
        save_tasks(htrain, f"{a.out}/hard_train_raw.jsonl")
        save_tasks(htest, f"{a.out}/hard_test_raw.jsonl")
        print("hard pairs (train):", Counter(tuple(t.meta["parts"]) for t in htrain).most_common(5))
        if a.validate:
            validate(htrain + htest)
        raise SystemExit
    train = make_split(a.train, seed=1234, prefix="train")
    test = make_split(a.test, seed=98765, prefix="test")
    save_tasks(train, f"{a.out}/train.jsonl")
    save_tasks(test, f"{a.out}/test.jsonl")
    print("train categories:", dict(Counter(t.category for t in train)))
    print("test categories:", dict(Counter(t.category for t in test)))
    if a.validate:
        validate(train + test)
