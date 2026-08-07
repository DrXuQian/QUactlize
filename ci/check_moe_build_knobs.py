#!/usr/bin/env python3
"""Exercise every advertised MoE sweep restriction against the real CMake generator."""
from __future__ import annotations

import concurrent.futures
import fnmatch
import os
import pathlib
import re
import shlex
import subprocess
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = ROOT / "dev/fold_derivation/gen_moe_units_check.sh"
BUILD = ROOT / "build.sh"
CMAKE_DRIVER = ROOT / "quactlize/csrc/CMakeLists.txt"
CMAKE_FRAGMENT = ROOT / "quactlize/csrc/CMakeLists.txt.in"
CMAKE_FILES = (CMAKE_DRIVER, CMAKE_FRAGMENT)
MOE_NAME = re.compile(r"\bMOE_[A-Z0-9_]+\b")
CACHE_TYPES = frozenset({"BOOL", "STRING", "PATH", "FILEPATH", "INTERNAL"})


@dataclass(frozen=True)
class CMakeCommand:
    name: str
    body: str
    start: int
    end: int


@dataclass(frozen=True)
class CMakeArgument:
    text: str
    quoted: bool


def cmake_commands(text: str):
    """Yield balanced CMake commands, including parentheses inside their quoted arguments.

    The advertised message itself contains ``(...)``. A non-greedy ``message(...)`` regex therefore stops before
    MOE_CORES and recreates the exact blind spot this check is meant to close. This small scanner skips comments,
    quoted strings and CMake bracket arguments while balancing the command's real parentheses.
    """
    size = len(text)
    cursor = 0
    while cursor < size:
        if text[cursor] == "#":
            bracket_comment = re.match(r"#\[(=*)\[", text[cursor:])
            if bracket_comment:
                close = "]" + bracket_comment.group(1) + "]"
                found = text.find(close, cursor + bracket_comment.end())
                cursor = size if found < 0 else found + len(close)
            else:
                newline = text.find("\n", cursor)
                cursor = size if newline < 0 else newline + 1
            continue
        if not (text[cursor].isalpha() or text[cursor] == "_"):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < size and (text[cursor].isalnum() or text[cursor] == "_"):
            cursor += 1
        name = text[start:cursor]
        opening = cursor
        while opening < size and text[opening].isspace():
            opening += 1
        if opening >= size or text[opening] != "(":
            continue

        depth = 1
        quote = False
        pos = opening + 1
        while pos < size and depth:
            char = text[pos]
            if quote:
                if char == "\\" and pos + 1 < size:
                    pos += 2
                    continue
                if char == '"':
                    quote = False
                pos += 1
                continue
            if char == '"':
                quote = True
                pos += 1
                continue
            if char == "#":
                newline = text.find("\n", pos)
                pos = size if newline < 0 else newline + 1
                continue
            bracket = re.match(r"\[(=*)\[", text[pos:])
            if bracket:
                close = "]" + bracket.group(1) + "]"
                found = text.find(close, pos + bracket.end())
                if found < 0:
                    raise ValueError(f"unterminated CMake bracket argument in {name}()")
                pos = found + len(close)
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            pos += 1
        if depth:
            raise ValueError(f"unterminated CMake command {name}()")
        yield CMakeCommand(name.lower(), text[opening + 1:pos - 1], start, pos)
        cursor = pos


def cmake_arguments(body: str) -> tuple[CMakeArgument, ...]:
    """Tokenize the subset of CMake arguments needed by message(), set(), and include()."""
    args: list[CMakeArgument] = []
    pos = 0
    size = len(body)
    while pos < size:
        while pos < size and body[pos].isspace():
            pos += 1
        if pos >= size:
            break
        if body[pos] == "#":
            bracket_comment = re.match(r"#\[(=*)\[", body[pos:])
            if bracket_comment:
                close = "]" + bracket_comment.group(1) + "]"
                found = body.find(close, pos + bracket_comment.end())
                if found < 0:
                    raise ValueError("unterminated CMake bracket comment")
                pos = found + len(close)
            else:
                newline = body.find("\n", pos)
                pos = size if newline < 0 else newline + 1
            continue
        if body[pos] == '"':
            pos += 1
            value: list[str] = []
            while pos < size and body[pos] != '"':
                if body[pos] == "\\" and pos + 1 < size:
                    value.append(body[pos + 1])
                    pos += 2
                else:
                    value.append(body[pos])
                    pos += 1
            if pos >= size:
                raise ValueError("unterminated quoted CMake argument")
            pos += 1
            args.append(CMakeArgument("".join(value), True))
            continue
        bracket = re.match(r"\[(=*)\[", body[pos:])
        if bracket:
            close = "]" + bracket.group(1) + "]"
            value_start = pos + bracket.end()
            found = body.find(close, value_start)
            if found < 0:
                raise ValueError("unterminated CMake bracket argument")
            args.append(CMakeArgument(body[value_start:found], True))
            pos = found + len(close)
            continue

        value = []
        while pos < size and not body[pos].isspace():
            if body[pos] == "#" and not value:
                break
            if body[pos] == "\\" and pos + 1 < size:
                value.append(body[pos + 1])
                pos += 2
            else:
                value.append(body[pos])
                pos += 1
        if value:
            args.append(CMakeArgument("".join(value), False))
    return tuple(args)


def _cache_name(command: CMakeCommand) -> str | None:
    if command.name != "set":
        return None
    args = cmake_arguments(command.body)
    if not args or not re.fullmatch(r"MOE_[A-Z0-9_]+", args[0].text):
        return None
    for index, arg in enumerate(args[1:-1], 1):
        if (not arg.quoted and arg.text.upper() == "CACHE"
                and not args[index + 1].quoted and args[index + 1].text.upper() in CACHE_TYPES):
            return args[0].text
    return None


def _advertised_names(command: CMakeCommand) -> tuple[frozenset[str], frozenset[str]]:
    if command.name != "message":
        return frozenset(), frozenset()
    args = cmake_arguments(command.body)
    if not args or args[0].text.upper() != "STATUS":
        return frozenset(), frozenset()
    rendered = "".join(arg.text for arg in args[1:])
    if not re.search(r"Narrow\s+a\s+MoE\s+axis", rendered):
        return frozenset(), frozenset()
    names = frozenset(MOE_NAME.findall(rendered))
    inner_close = rendered.find(")")
    tail = frozenset(MOE_NAME.findall(rendered[inner_close + 1:])) if inner_close >= 0 else frozenset()
    return names, tail


@dataclass(frozen=True)
class ForwardLoop:
    iterator: str
    array: str
    names: frozenset[str]
    values_start: int
    values_end: int
    array_start: int
    array_end: int
    append_end: int
    loop_end: int


@dataclass(frozen=True)
class ForwardSink:
    array: str
    array_start: int
    array_end: int
    command_start: int


@dataclass(frozen=True)
class ShellWord:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class BuildForwarding:
    loops: tuple[ForwardLoop, ...]
    sinks: tuple[ForwardSink, ...]
    sink_arrays: frozenset[str]
    connected_arrays: frozenset[str]
    loop_names: frozenset[str]
    forwarded: frozenset[str]
    literal_cache_controls: frozenset[str]


def _shell_comment_start(text: str, pos: int) -> bool:
    return (text[pos] == "#"
            and (pos == 0 or text[pos - 1].isspace() or text[pos - 1] in ";|&()"))


def _shell_code(text: str, *, keep_double_contents: bool) -> str:
    """Mask shell comments/quotes/escapes without changing offsets.

    With keep_double_contents, active parameter expansions remain visible for argv provenance. With it off, only
    shell structure remains, so prose inside a quoted argument cannot masquerade as a loop, assignment, or unset.
    Backslash-newline is replaced by spaces in both modes so a continued command becomes one logical line.
    """
    out = list(text)
    pos = 0
    state = "plain"
    while pos < len(text):
        char = text[pos]
        if state == "plain":
            if _shell_comment_start(text, pos):
                newline = text.find("\n", pos)
                end = len(text) if newline < 0 else newline
                out[pos:end] = " " * (end - pos)
                pos = end
                continue
            if char == "'":
                out[pos] = " "
                state = "single"
                pos += 1
                continue
            if char == '"':
                out[pos] = " "
                state = "double"
                pos += 1
                continue
            if char == "\\" and pos + 1 < len(text):
                out[pos] = " "
                out[pos + 1] = " "
                pos += 2
                continue
            pos += 1
            continue
        if state == "single":
            if char == "'":
                state = "plain"
            out[pos] = " "
            pos += 1
            continue
        if char == '"':
            out[pos] = " "
            state = "plain"
            pos += 1
            continue
        if char == "\\" and pos + 1 < len(text) and text[pos + 1] in '$`"\\\n':
            out[pos] = " "
            out[pos + 1] = " "
            pos += 2
            continue
        if not keep_double_contents or char in "\r\n":
            out[pos] = " "
        pos += 1
    if state != "plain":
        raise ValueError(f"unterminated {state}-quoted shell string")
    return "".join(out)


def _shell_logical_commands(text: str):
    """Yield offset-preserving logical lines after continuations have already been masked."""
    offset = 0
    for line in text.splitlines(keepends=True):
        start = offset
        offset += len(line)
        yield start, offset, line
    if offset < len(text):
        yield offset, len(text), text[offset:]


def _shell_words(command: str, base_start: int) -> tuple[ShellWord, ...]:
    """Return argv-like shell words up to the first control/redirection operator.

    Quote delimiters and continuation backslashes do not become part of a word, but quoted whitespace does. This is
    deliberately a fail-closed tokenizer for the simple configure command in build.sh, not a general Bash parser.
    """
    words: list[ShellWord] = []
    value: list[str] = []
    word_start: int | None = None
    state = "plain"
    pos = 0

    def finish(end: int) -> None:
        nonlocal value, word_start
        if word_start is not None:
            words.append(ShellWord("".join(value), base_start + word_start, base_start + end))
            value = []
            word_start = None

    while pos < len(command):
        char = command[pos]
        if state == "plain":
            if _shell_comment_start(command, pos):
                finish(pos)
                break
            if char.isspace():
                finish(pos)
                pos += 1
                continue
            if char in ";|&<>()":
                finish(pos)
                break
            if word_start is None:
                word_start = pos
            if char == "'":
                state = "single"
                pos += 1
                continue
            if char == '"':
                state = "double"
                pos += 1
                continue
            if char == "\\" and pos + 1 < len(command):
                if command[pos + 1] not in "\r\n":
                    value.append(command[pos + 1])
                pos += 2
                continue
            value.append(char)
            pos += 1
            continue
        if state == "single":
            if char == "'":
                state = "plain"
            else:
                value.append(char)
            pos += 1
            continue
        if char == '"':
            state = "plain"
            pos += 1
            continue
        if char == "\\" and pos + 1 < len(command) and command[pos + 1] in '$`"\\\r\n':
            if command[pos + 1] not in "\r\n":
                value.append(command[pos + 1])
            pos += 2
            continue
        value.append(char)
        pos += 1
    finish(pos)
    if state != "plain":
        raise ValueError(f"unterminated {state}-quoted shell word")
    return tuple(words)


def _is_configure_cmake(words: tuple[ShellWord, ...]) -> bool:
    values = [word.text for word in words]
    if values[:1] == ["command"]:
        values = values[1:]
    if not values or values[0] != "cmake":
        return False
    non_configure = {"-E", "-P", "-N", "--build", "--install", "--open", "--find-package", "--workflow",
                     "--version", "--system-information", "--print-config-dir", "--list-presets"}
    has_configure_anchor = False
    for value in values[1:]:
        if (value in non_configure or value.startswith("-P") or value.startswith("--build=")
                or value.startswith("--install=") or value.startswith("--help")
                or value.startswith("--list-presets=") or re.fullmatch(r"-L[AHT]*", value)):
            return False
        if (value.startswith("-S") and value != "-S") or value in {"-S", "--preset"}:
            has_configure_anchor = True
        elif not value.startswith("-") and not re.fullmatch(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\](?:\+\$\{\1\[@\]\})?\}", value):
            has_configure_anchor = True
    return has_configure_anchor


def _literal_cache_controls(words: tuple[ShellWord, ...], known_names: set[str]) -> set[str]:
    controls: set[str] = set()
    values = [word.text for word in words]
    for index, value in enumerate(values):
        candidates = [value]
        if value in {"-D", "-U"} and index + 1 < len(values):
            candidates.append(value + values[index + 1])
        for candidate in candidates:
            match = re.fullmatch(r"-D(MOE_[A-Z0-9_]+)(?::[^=]+)?=.*", candidate)
            if match:
                controls.add(match.group(1))
                continue
            unset = re.fullmatch(r"-U(.+)", candidate)
            if unset:
                controls.update(name for name in known_names if fnmatch.fnmatchcase(name, unset.group(1)))
    return controls


def _has_array_reset(structure: str, start: int, end: int, array: str) -> bool:
    # This forwarding idiom has no legitimate intermediate use: the loop fills the array and the configure command
    # consumes it. Fail closed on any unquoted structural mention in between. That covers direct assignment/unset as
    # well as mutators such as mapfile/read without pretending to implement Bash data-flow analysis.
    return bool(re.search(rf"\b{re.escape(array)}\b", structure[start:end]))


def build_forwarding(build_text: str) -> BuildForwarding:
    """Derive F from a literal loop through a live array into an active cmake argv expansion."""
    active = _shell_code(build_text, keep_double_contents=True)
    structure = _shell_code(build_text, keep_double_contents=False)
    loops: list[ForwardLoop] = []
    loop_pattern = re.compile(
        r"(?m)^[ \t]*for[ \t]+(?P<iterator>[A-Za-z_][A-Za-z0-9_]*)[ \t]+in[ \t]*"
        r"(?P<values>[^;\r\n]*?)(?:[ \t]*;[ \t]*|[ \t]*\r?\n[ \t]*)do\b"
        r"(?P<body>.*?)^[ \t]*done\b", re.S
    )
    for match in loop_pattern.finditer(structure):
        iterator = match.group("iterator")
        iterator_ref = rf"(?:\${re.escape(iterator)}|\$\{{{re.escape(iterator)}\}})"
        indirect_ref = rf"\$\{{!{re.escape(iterator)}\}}"
        append_pattern = re.compile(
            rf"(?P<array>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*\(\s*-D"
            rf"{iterator_ref}={indirect_ref}\s*\)", re.S
        )
        raw_values = re.sub(r"\\\r?\n", " ", build_text[match.start("values"):match.end("values")])
        try:
            words = shlex.split(raw_values, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"cannot parse values for build.sh loop {iterator}: {exc}") from exc
        names = frozenset(word for word in words if re.fullmatch(r"MOE_[A-Z0-9_]+", word))
        body_start = match.start("body")
        body_active = active[body_start:match.end("body")]
        for append in append_pattern.finditer(body_active):
            array_start = body_start + append.start("array")
            # The exact append text may contain active expansions inside double quotes, but its assignment target must
            # be shell syntax, not prose in an echo string.
            if structure[array_start] == " ":
                continue
            array_end = body_start + append.end("array")
            mentions = {(body_start + mention.start(), body_start + mention.end())
                        for mention in re.finditer(rf"\b{re.escape(append.group('array'))}\b",
                                                   structure[body_start:match.end("body")])}
            # A mutation anywhere else in the loop body also runs on the next iteration. Accept only the append target
            # itself; aliases or extra reads/mutators require a real shell parser and therefore fail closed here.
            if mentions != {(array_start, array_end)}:
                continue
            append_expr = (
                rf"{re.escape(append.group('array'))}\s*\+=\s*\(\s*-D"
                rf"{iterator_ref}={indirect_ref}\s*\)"
            )
            minimal_body = rf"\s*{append_expr}\s*;?\s*"
            canonical_body = (
                rf"\s*if\s+\[\s+-n\s+\$\{{!{re.escape(iterator)}:-\}}\s+\]\s*;\s*then\s+"
                rf"{append_expr}\s*;\s*echo\s+\[build\.sh\]\s+{iterator_ref}={indirect_ref}"
                rf"\s*;\s*fi\s*"
            )
            if not (re.fullmatch(minimal_body, body_active, re.S)
                    or re.fullmatch(canonical_body, body_active, re.S)):
                continue
            loops.append(ForwardLoop(
                iterator, append.group("array"), names, match.start("values"), match.end("values"),
                array_start, array_end, body_start + append.end(), match.end(),
            ))

    loop_names = {name for loop in loops for name in loop.names}
    sinks: list[ForwardSink] = []
    literal_controls: set[str] = set()
    array_ref = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]")
    for start, end, _ in _shell_logical_commands(active):
        original_command = build_text[start:end]
        words = _shell_words(original_command, start)
        if not _is_configure_cmake(words):
            continue
        # A second configure invocation can overwrite or unset the cache populated by the forwarding invocation, so
        # literal controls are forbidden across the whole build route, not only on the line containing the array.
        literal_controls.update(_literal_cache_controls(words, loop_names))
        command_sinks: list[ForwardSink] = []
        for word in words:
            active_word = active[word.start:word.end]
            refs = tuple(array_ref.finditer(active_word))
            for array in sorted({ref.group(1) for ref in refs}):
                simple = f"${{{array}[@]}}"
                conditional = f"${{{array}[@]+${{{array}[@]}}}}"
                if word.text not in {simple, conditional}:
                    continue
                for ref in refs:
                    if ref.group(1) == array:
                        command_sinks.append(
                            ForwardSink(array, word.start + ref.start(1), word.start + ref.end(1), start)
                        )
        if command_sinks:
            sinks.extend(command_sinks)

    connected_arrays: set[str] = set()
    forwarded: set[str] = set()
    for loop in loops:
        live_sinks = [sink for sink in sinks if sink.array == loop.array and sink.command_start > loop.append_end
                      and not _has_array_reset(structure, loop.append_end, sink.command_start, loop.array)]
        if live_sinks:
            connected_arrays.add(loop.array)
            forwarded.update(loop.names)
    sink_arrays = {sink.array for sink in sinks}
    return BuildForwarding(tuple(loops), tuple(sinks), frozenset(sink_arrays), frozenset(connected_arrays),
                           frozenset(loop_names), frozenset(forwarded), frozenset(literal_controls))


@dataclass(frozen=True)
class BuildInterface:
    advertised: frozenset[str]
    advertised_after_inner_close: frozenset[str]
    forwarded: frozenset[str]
    caches: frozenset[str]
    build: BuildForwarding
    active_fragment_includes: int


def _active_fragment_include_count(driver_text: str) -> int:
    expected = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{CMAKE_FRAGMENT.name}"
    count = 0
    for command in cmake_commands(driver_text):
        if command.name != "include":
            continue
        args = cmake_arguments(command.body)
        if len(args) == 1 and args[0].text == expected:
            count += 1
    return count


def derive_build_interface(cmake_texts: dict[pathlib.Path, str], build_text: str) -> BuildInterface:
    advertised: set[str] = set()
    tail_advertised: set[str] = set()
    caches: set[str] = set()
    for text in cmake_texts.values():
        for command in cmake_commands(text):
            cache = _cache_name(command)
            if cache:
                caches.add(cache)
            names, tail = _advertised_names(command)
            advertised.update(names)
            tail_advertised.update(tail)
    forwarding = build_forwarding(build_text)
    return BuildInterface(frozenset(advertised), frozenset(tail_advertised), forwarding.forwarded,
                          frozenset(caches), forwarding,
                          _active_fragment_include_count(cmake_texts.get(CMAKE_DRIVER, "")))


def interface_problems(interface: BuildInterface) -> list[str]:
    problems = []
    if interface.active_fragment_includes != 1:
        problems.append(
            f"{CMAKE_FRAGMENT.name} has {interface.active_fragment_includes} active includes from "
            f"{CMAKE_DRIVER.name}, expected 1"
        )
    if not interface.advertised:
        problems.append("the Narrow-a-MoE-axis message advertised no MOE_* names")
    if not interface.forwarded:
        problems.append("no -D forwarding array reaches a cmake command")
    if not interface.caches:
        problems.append("CMake declares no MOE_* CACHE variables")
    missing_forward = interface.advertised - interface.forwarded
    missing_cache = interface.forwarded - interface.caches
    literal_conflicts = interface.build.loop_names & interface.build.literal_cache_controls
    if missing_forward:
        problems.append("advertised but not forwarded: " + ", ".join(sorted(missing_forward)))
    if missing_cache:
        problems.append("forwarded without a CACHE declaration: " + ", ".join(sorted(missing_cache)))
    if literal_conflicts:
        problems.append("forwarded names also have literal cmake -D/-U controls: "
                        + ", ".join(sorted(literal_conflicts)))
    return problems


def _remove_forwarded_name(build_text: str, interface: BuildInterface, victim: str) -> str:
    spans = {(loop.values_start, loop.values_end) for loop in interface.build.loops
             if loop.array in interface.build.connected_arrays and victim in loop.names}
    if not spans:
        raise ValueError(f"cannot plant loop removal for {victim}")
    planted = build_text
    removed = 0
    for start, end in sorted(spans, reverse=True):
        values, count = re.subn(rf"\b{re.escape(victim)}\b", "", planted[start:end])
        planted = planted[:start] + values + planted[end:]
        removed += count
    if not removed:
        raise ValueError(f"loop-removal plant did not change {victim}")
    return planted


def _redirect_forward_appends(build_text: str, interface: BuildInterface) -> str:
    spans = {(loop.array_start, loop.array_end) for loop in interface.build.loops
             if loop.array in interface.build.connected_arrays}
    if not spans:
        raise ValueError("cannot plant a disconnected forwarding append")
    planted = build_text
    for start, end in sorted(spans, reverse=True):
        planted = planted[:start] + planted[start:end] + "_DISCONNECTED" + planted[end:]
    return planted


def _remove_cache(cmake_texts: dict[pathlib.Path, str], victim: str) -> dict[pathlib.Path, str]:
    planted = dict(cmake_texts)
    removed = 0
    for path, text in cmake_texts.items():
        spans = [(command.start, command.end) for command in cmake_commands(text)
                 if _cache_name(command) == victim]
        for start, end in reversed(spans):
            blank = "".join("\n" if char == "\n" else " " for char in planted[path][start:end])
            planted[path] = planted[path][:start] + blank + planted[path][end:]
            removed += 1
    if removed != 1:
        raise ValueError(f"expected one CACHE declaration for {victim}, removed {removed}")
    return planted


def _disconnect_forward_sink(build_text: str, interface: BuildInterface) -> str:
    planted = build_text
    spans = {(sink.array_start, sink.array_end) for sink in interface.build.sinks
             if sink.array in interface.build.connected_arrays}
    if not spans:
        raise ValueError("cannot plant a disconnected cmake forwarding sink")
    for start, end in sorted(spans, reverse=True):
        planted = planted[:start] + planted[start:end] + "_DISCONNECTED" + planted[end:]
    return planted


def _reset_forward_arrays(build_text: str, interface: BuildInterface) -> str:
    live_loops = [loop for loop in interface.build.loops if loop.array in interface.build.connected_arrays]
    live_sinks = [sink for sink in interface.build.sinks if sink.array in interface.build.connected_arrays]
    if not live_loops or not live_sinks:
        raise ValueError("cannot plant a forwarding-array reset")
    insertion = max(loop.loop_end for loop in live_loops)
    first_sink = min(sink.command_start for sink in live_sinks)
    if insertion >= first_sink:
        raise ValueError("cannot place forwarding-array reset between append and cmake sink")
    reset = "\n" + "\n".join(f"{array}=() # dynamic reset plant" for array in sorted(interface.build.connected_arrays))
    return build_text[:insertion] + reset + build_text[insertion:]


def _add_literal_cache_control(build_text: str, interface: BuildInterface, victim: str) -> str:
    command_starts = {sink.command_start for sink in interface.build.sinks
                      if sink.array in interface.build.connected_arrays}
    if len(command_starts) != 1:
        raise ValueError(f"cannot plant a literal cache override across {len(command_starts)} configure commands")
    target = next(iter(command_starts))
    active = _shell_code(build_text, keep_double_contents=True)
    commands = [(start, end) for start, end, _ in _shell_logical_commands(active) if start == target]
    if len(commands) != 1:
        raise ValueError("cannot locate configure command for literal cache override")
    start, end = commands[0]
    words = _shell_words(build_text[start:end], start)
    if not _is_configure_cmake(words) or not words:
        raise ValueError("literal cache override target is not a configure command")
    insertion = words[-1].end
    return build_text[:insertion] + f" -D{victim}=PLANTED" + build_text[insertion:]


def _add_advertised_name(cmake_texts: dict[pathlib.Path, str], fresh: str) -> dict[pathlib.Path, str]:
    planted = dict(cmake_texts)
    hits: list[tuple[pathlib.Path, CMakeCommand]] = []
    for path, text in cmake_texts.items():
        hits.extend((path, command) for command in cmake_commands(text) if _advertised_names(command)[0])
    if len(hits) != 1:
        raise ValueError(f"expected one advertised MoE message, found {len(hits)}")
    path, command = hits[0]
    insertion = command.end - 1
    planted[path] = planted[path][:insertion] + f' " / {fresh}"' + planted[path][insertion:]
    return planted


def _disconnect_fragment_include(cmake_texts: dict[pathlib.Path, str]) -> dict[pathlib.Path, str]:
    planted = dict(cmake_texts)
    text = planted[CMAKE_DRIVER]
    expected = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{CMAKE_FRAGMENT.name}"
    hits = [command for command in cmake_commands(text)
            if command.name == "include" and [arg.text for arg in cmake_arguments(command.body)] == [expected]]
    if len(hits) != 1:
        raise ValueError(f"cannot plant active-include disconnection: found {len(hits)} includes")
    command = hits[0]
    planted[CMAKE_DRIVER] = text[:command.start] + "qz_skip" + text[command.start + len("include"):]
    return planted


def _expect_case(label: str, interface: BuildInterface, *, advertised: frozenset[str],
                 forwarded: frozenset[str], caches: frozenset[str], loop_names: frozenset[str],
                 include_count: int, problems: list[str]) -> None:
    observed = (interface.advertised, interface.forwarded, interface.caches, interface.build.loop_names,
                interface.active_fragment_includes, interface_problems(interface))
    expected = (advertised, forwarded, caches, loop_names, include_count, problems)
    if observed != expected:
        raise ValueError(f"{label} control mismatch: observed={observed!r}, expected={expected!r}")


def _missing_forward_problem(names: frozenset[str]) -> str:
    return "advertised but not forwarded: " + ", ".join(sorted(names))


def _missing_cache_problem(names: frozenset[str]) -> str:
    return "forwarded without a CACHE declaration: " + ", ".join(sorted(names))


def _check_parser_layout_controls() -> None:
    shell = r'''for knob in MOE_ALPHA \
  MOE_BETA
do
  ROUTE+=(
    "-D${knob}=${!knob}"
  )
done
cmake source "${ROUTE[@]}"
'''
    expected = frozenset({"MOE_ALPHA", "MOE_BETA"})
    if build_forwarding(shell).forwarded != expected:
        raise ValueError("multiline for/append/cmake positive control did not derive F")
    dead_sinks = (
        "cmake source '${ROUTE[@]}'",
        r"cmake source \${ROUTE[@]}",
        "cmake source # ${ROUTE[@]}",
        'cmake source "prefix${ROUTE[@]}suffix"',
        'cmake source "-DDEAD=${ROUTE[@]}"',
        'cmake -E echo "${ROUTE[@]}"',
        'cmake -P script.cmake "${ROUTE[@]}"',
        'cmake --build build "${ROUTE[@]}"',
        'cmake --list-presets=all "${ROUTE[@]}"',
        'cmake --print-config-dir "${ROUTE[@]}"',
        'cmake -N "${ROUTE[@]}"',
        'cmake -LA "${ROUTE[@]}"',
    )
    for sink in dead_sinks:
        planted = shell.replace('cmake source "${ROUTE[@]}"', sink)
        if build_forwarding(planted).forwarded:
            raise ValueError(f"inactive cmake argv expansion was accepted: {sink!r}")
    for mutation in ("ROUTE=()", "unset ROUTE", "mapfile -t ROUTE </dev/null"):
        if build_forwarding(shell.replace("cmake source", mutation + "\ncmake source")).forwarded:
            raise ValueError(f"an array mutation between append and cmake sink was accepted: {mutation}")
        in_loop = shell.replace('  ROUTE+=(\n', f'  {mutation}\n  ROUTE+=(\n')
        if build_forwarding(in_loop).forwarded:
            raise ValueError(f"an array mutation in the forwarding loop was accepted: {mutation}")
    filtered = shell.replace('  ROUTE+=(\n', '  [ "$knob" = MOE_ALPHA ] && continue\n  ROUTE+=(\n')
    if build_forwarding(filtered).forwarded:
        raise ValueError("a victim-filtered forwarding loop was accepted")
    subshell = shell.replace('  ROUTE+=(\n', '  ( ROUTE+=(\n').replace('  )\ndone', '  ) )\ndone')
    if build_forwarding(subshell).forwarded:
        raise ValueError("a subshell-only forwarding append was accepted")
    if build_forwarding(shell.replace("-D${knob}", "-X${knob}")).forwarded:
        raise ValueError("a non--D array append was accepted as forwarding")
    literal = build_forwarding(shell.replace('cmake source "${ROUTE[@]}"',
                                              'cmake source "${ROUTE[@]}" -DMOE_ALPHA=planted -U MOE_BETA'))
    if literal.literal_cache_controls != expected:
        raise ValueError(f"literal cmake -D/-U controls were not derived: {literal.literal_cache_controls}")
    wildcard_unset = build_forwarding(shell.replace('cmake source "${ROUTE[@]}"',
                                                     'cmake source "${ROUTE[@]}" -U "MOE_*"'))
    if wildcard_unset.literal_cache_controls != expected:
        raise ValueError(f"literal cmake -U glob was not expanded over F: {wildcard_unset.literal_cache_controls}")
    second_configure = build_forwarding(shell + 'cmake source -DMOE_ALPHA=planted -U "MOE_BETA"\n')
    if second_configure.literal_cache_controls != expected:
        raise ValueError("a second configure command's literal cache controls were missed")

    cmake = '''message(STATUS "Narrow a MoE axis (" "MOE_ALPHA); " "MOE_BETA")
set(
  MOE_ALPHA ""
  CACHE
  STRING
  "real cache"
)
set(MOE_BETA "not a CACHE STRING declaration")
set(MOE_GAMMA "" # CACHE STRING is only a comment
)
'''
    commands = tuple(cmake_commands(cmake))
    advertised = set()
    caches = set()
    for command in commands:
        advertised.update(_advertised_names(command)[0])
        cache = _cache_name(command)
        if cache:
            caches.add(cache)
    if advertised != expected or caches != {"MOE_ALPHA"}:
        raise ValueError(f"CMake parser controls failed: A={advertised}, C={caches}")


def check_build_interface() -> tuple[BuildInterface, str]:
    cmake_texts = {path: path.read_text(errors="replace") for path in CMAKE_FILES}
    build_text = BUILD.read_text(errors="replace")
    _check_parser_layout_controls()
    baseline = derive_build_interface(cmake_texts, build_text)
    baseline_problems = interface_problems(baseline)
    if baseline_problems:
        raise ValueError("baseline forwarding contract is not clean: " + "; ".join(baseline_problems))
    if not baseline.advertised or not baseline.forwarded or not baseline.caches:
        raise ValueError("baseline A/F/C sets must all be nonempty")

    candidates = baseline.advertised & baseline.forwarded & baseline.caches
    tail_candidates = candidates & baseline.advertised_after_inner_close
    if not candidates:
        raise ValueError("baseline A/F/C intersection is empty; no dynamic victim exists")
    victim = sorted(tail_candidates or candidates)[0]

    # Exercise every derived member, including the exact MOE_STAGES omission that motivated this gate. No control
    # relies on a copied knob list, and the tail-preferred victim below separately proves the message's inner `)` did
    # not truncate A before MOE_CORES.
    for candidate in sorted(candidates):
        missing_loop = derive_build_interface(
            cmake_texts, _remove_forwarded_name(build_text, baseline, candidate)
        )
        _expect_case(
            f"remove loop victim {candidate}", missing_loop, advertised=baseline.advertised,
            forwarded=baseline.forwarded - {candidate}, caches=baseline.caches,
            loop_names=baseline.build.loop_names - {candidate}, include_count=1,
            problems=[_missing_forward_problem(frozenset({candidate}))],
        )

        missing_cache = derive_build_interface(_remove_cache(cmake_texts, candidate), build_text)
        _expect_case(
            f"remove CACHE victim {candidate}", missing_cache, advertised=baseline.advertised,
            forwarded=baseline.forwarded, caches=baseline.caches - {candidate},
            loop_names=baseline.build.loop_names, include_count=1,
            problems=[_missing_cache_problem(frozenset({candidate}))],
        )

    disconnected_append = derive_build_interface(cmake_texts, _redirect_forward_appends(build_text, baseline))
    disconnected_problems = ["no -D forwarding array reaches a cmake command",
                             _missing_forward_problem(baseline.advertised)]
    _expect_case(
        "disconnect append array", disconnected_append, advertised=baseline.advertised,
        forwarded=frozenset(), caches=baseline.caches, loop_names=baseline.build.loop_names,
        include_count=1, problems=disconnected_problems,
    )

    disconnected = derive_build_interface(cmake_texts, _disconnect_forward_sink(build_text, baseline))
    _expect_case(
        "disconnect cmake sink", disconnected, advertised=baseline.advertised,
        forwarded=frozenset(), caches=baseline.caches, loop_names=baseline.build.loop_names,
        include_count=1, problems=disconnected_problems,
    )

    reset = derive_build_interface(cmake_texts, _reset_forward_arrays(build_text, baseline))
    _expect_case(
        "reset forwarding array", reset, advertised=baseline.advertised,
        forwarded=frozenset(), caches=baseline.caches, loop_names=baseline.build.loop_names,
        include_count=1, problems=disconnected_problems,
    )

    literal_override = derive_build_interface(
        cmake_texts, _add_literal_cache_control(build_text, baseline, victim)
    )
    _expect_case(
        "add literal cache override", literal_override, advertised=baseline.advertised,
        forwarded=baseline.forwarded, caches=baseline.caches, loop_names=baseline.build.loop_names,
        include_count=1,
        problems=[f"forwarded names also have literal cmake -D/-U controls: {victim}"],
    )

    occupied = baseline.advertised | baseline.forwarded | baseline.caches | baseline.build.loop_names
    fresh = "MOE_GATE_CONTROL"
    while fresh in occupied or any(fresh in text for text in cmake_texts.values()):
        fresh += "_X"
    fresh_advertised = derive_build_interface(_add_advertised_name(cmake_texts, fresh), build_text)
    _expect_case(
        "add advertised name", fresh_advertised, advertised=baseline.advertised | {fresh},
        forwarded=baseline.forwarded, caches=baseline.caches, loop_names=baseline.build.loop_names,
        include_count=1, problems=[_missing_forward_problem(frozenset({fresh}))],
    )

    inactive_fragment = derive_build_interface(_disconnect_fragment_include(cmake_texts), build_text)
    _expect_case(
        "disconnect active CMake fragment", inactive_fragment, advertised=baseline.advertised,
        forwarded=baseline.forwarded, caches=baseline.caches, loop_names=baseline.build.loop_names,
        include_count=0,
        problems=[f"{CMAKE_FRAGMENT.name} has 0 active includes from {CMAKE_DRIVER.name}, expected 1"],
    )
    return baseline, victim


POSITIVE = {
    "default": {},
    "format": {"MOE_FORMATS": "i4"},
    "tile-m": {"MOE_TM_LIST": "16"},
    "tile-n": {"MOE_TN_LIST": "32"},
    "warp-m": {"MOE_WM_LIST": "16"},
    "stages": {"MOE_STAGES": "12"},
}
NEGATIVE_CONTROLS = {
    "tile-filter-bypass": {"BAD": "5", "MOE_TM_LIST": "16"},
    "stage-device-flag-drop": {"BAD": "6", "MOE_STAGES": "12"},
}
LEGACY_POSITIVE = {
    "legacy-stage": {"PPU_DEFS": "MOE_STAGES_4"},
}
INVALID = {
    "partial-format-typo": {"MOE_FORMATS": "i4;typo"},
    "bad-tile-m": {"MOE_TM_LIST": "17"},
    "bad-tile-n": {"MOE_TN_LIST": "17"},
    "bad-warp-m": {"MOE_WM_LIST": "17"},
    "bad-stage": {"MOE_STAGES": "5"},
}
EXPECTED_FAILURES = {
    "legacy-stage-conflict": (
        {"PPU_DEFS": "MOE_STAGES_4", "MOE_STAGES": "2"},
        "both select the stage axis",
    ),
    "unknown-legacy-stage": (
        {"PPU_DEFS": "MOE_STAGES_5"},
        "unknown legacy stage selector",
    ),
    "global-filter-empties-decode": (
        {"MOE_TM_LIST": "64"},
        "decode sweep produced no legal shapes",
    ),
    "zero-cores-before-use": (
        {"MOE_CHECK_CORES": "0"},
        "MOE_CORES must be a positive integer",
    ),
}


def run_case(payload):
    item, derived_knobs = payload
    name, extra = item
    env = os.environ.copy()
    for key in {"BAD", "PPU_DEFS", "MOE_CHECK_CORES", *derived_knobs}:
        env.pop(key, None)
    # The behavior half must exercise the same active fragment inspected by A/C, regardless of the caller's ambient
    # environment. Its MOE_* scrub list comes from A/F/C, so a newly advertised knob cannot inherit a hidden value.
    env["QUACTLIZE_CMAKE"] = str(CMAKE_FRAGMENT)
    env.update(extra)
    result = subprocess.run(["bash", str(GEN)], cwd=ROOT, env=env, capture_output=True, text=True)
    return name, result.returncode, result.stdout + result.stderr


def counts(log: str):
    found = {}
    for band in ("full", "decode"):
        match = re.search(rf"-- OK {band}: ([0-9]+) shapes", log)
        if not match:
            raise ValueError(f"no independently-checked {band} shape count")
        found[band] = int(match.group(1))
    return found


def main() -> int:
    required = (GEN, BUILD, *CMAKE_FILES)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print(f"[moe-build-knobs] ERROR: missing {', '.join(missing)}")
        return 1
    try:
        interface, interface_victim = check_build_interface()
    except (OSError, ValueError) as exc:
        print(f"[moe-build-knobs] ERROR: build.sh/CMake forwarding contract: {exc}")
        return 1
    expected_failure_envs = {name: spec[0] for name, spec in EXPECTED_FAILURES.items()}
    work = (list(POSITIVE.items()) + list(LEGACY_POSITIVE.items()) + list(NEGATIVE_CONTROLS.items())
            + list(INVALID.items()) + list(expected_failure_envs.items()))
    derived_knobs = interface.advertised | interface.forwarded | interface.caches
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        payloads = ((item, derived_knobs) for item in work)
        results = dict((name, (rc, log)) for name, rc, log in pool.map(run_case, payloads))

    failures = []
    observed = {}
    for name in POSITIVE:
        rc, log = results[name]
        if rc != 0:
            failures.append(f"{name}: expected pass, exit={rc}: {log.strip().splitlines()[-1:]}")
            continue
        try:
            observed[name] = counts(log)
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
    if "default" in observed:
        base = observed["default"]
        for name, got in observed.items():
            if name == "default":
                continue
            if not any(got[band] < base[band] for band in ("full", "decode")):
                failures.append(f"{name}: restriction did not shrink either band: {got} vs {base}")

    for name in NEGATIVE_CONTROLS:
        rc, log = results[name]
        if rc != 0 or "the generator was REJECTED" not in log:
            failures.append(f"{name}: planted defect was not cleanly rejected (exit={rc})")
    for name in INVALID:
        rc, log = results[name]
        if rc == 0 or "expected a semicolon-separated subset" not in log:
            failures.append(f"{name}: invalid/partly-invalid value was not rejected by validation (exit={rc})")
    for name in LEGACY_POSITIVE:
        rc, log = results[name]
        if rc != 0:
            failures.append(f"{name}: supported legacy selector failed (exit={rc})")
    for name, (_, expected) in EXPECTED_FAILURES.items():
        rc, log = results[name]
        if rc == 0 or expected not in log:
            failures.append(f"{name}: expected failure containing {expected!r}, exit={rc}")

    if failures:
        print(f"[moe-build-knobs] FAIL: {len(failures)} problem(s)")
        for failure in failures:
            print(f"    {failure}")
        return 1
    summary = ", ".join(f"{name}={v['full']}/{v['decode']}" for name, v in observed.items())
    advertised = ",".join(sorted(interface.advertised))
    forwarded = ",".join(sorted(interface.forwarded))
    caches = ",".join(sorted(interface.caches))
    print(f"[moe-build-knobs] PASS: A={advertised}; F={forwarded}; C={caches}; "
          f"dynamic victim {interface_victim} caught loop/cache/append/sink/reset/literal defects; "
          f"standalone-configure-argv, fresh-advertisement, and active-include controls passed; {summary}; "
          "legacy positive passed, two generator plants and nine invalid/policy inputs rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
