"""Structural checking of generated TSX: balanced brackets, closed strings, matching JSX tags.

This is not a compiler, and it does not check types or names. It exists because the critic must
catch unclosed markup without a JavaScript toolchain in the process. It tracks context explicitly,
because ``don't`` is harmless in JSX text but opens a string in code. It uses an explicit stack, so
hostile nesting cannot exhaust the call stack.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

_CLOSERS: Final = {")": "(", "]": "[", "}": "{"}
_EXPRESSION_KEYWORDS: Final = frozenset(
    {"return", "default", "yield", "case", "else", "do", "await", "typeof", "void", "in", "of"}
)
# After these, "<" starts markup. After a name or a closing bracket it is a comparison or a generic.
_EXPRESSION_PUNCTUATION: Final = frozenset("(,=[{:;?&|!>+-*%~^")
_NAME_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$.:-"
)
_WORD_CHARS: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$")


@dataclass(frozen=True, slots=True)
class TsxIssue:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


class _Mode(Enum):
    CODE = auto()
    TEMPLATE = auto()
    TAG = auto()
    CHILDREN = auto()


@dataclass(slots=True)
class _Frame:
    mode: _Mode
    label: str
    line: int


class _Failure(Exception):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(message)
        self.issue = TsxIssue(line, message)


class _Scanner:
    def __init__(self, code: str) -> None:
        self.code = code
        self.position = 0
        self.stack = [_Frame(_Mode.CODE, "", 1)]
        self.previous = ""
        self.previous_word = ""

    @property
    def line(self) -> int:
        return self.code.count("\n", 0, self.position) + 1

    def run(self) -> None:
        handlers = {
            _Mode.CODE: self._code,
            _Mode.TEMPLATE: self._template,
            _Mode.TAG: self._tag,
            _Mode.CHILDREN: self._children,
        }
        while self.position < len(self.code):
            handlers[self.stack[-1].mode]()
        frame = self.stack[-1]
        if len(self.stack) > 1:
            raise _Failure(frame.line, f"{frame.label} is never closed")

    def _startswith(self, text: str) -> bool:
        return self.code.startswith(text, self.position)

    def _peek(self, offset: int = 1) -> str:
        return self.code[self.position + offset : self.position + offset + 1]

    def _note(self, char: str) -> None:
        self.previous = char
        self.previous_word = ""

    def _code(self) -> None:
        char = self.code[self.position]
        if char.isspace():
            self.position += 1
        elif self._startswith("//"):
            end = self.code.find("\n", self.position)
            self.position = len(self.code) if end == -1 else end
        elif self._startswith("/*"):
            end = self.code.find("*/", self.position + 2)
            if end == -1:
                raise _Failure(self.line, "block comment is never closed")
            self.position = end + 2
        elif char in {"'", '"'}:
            self._string(char)
            self._note(char)
        elif char == "`":
            self.stack.append(_Frame(_Mode.TEMPLATE, "template literal", self.line))
            self.position += 1
        elif char in "([{":
            self.stack.append(_Frame(_Mode.CODE, char, self.line))
            self.position += 1
            self._note(char)
        elif char in _CLOSERS:
            self._close_bracket(char)
        elif char == "<" and self._markup_can_start():
            self._open_markup()
        elif char in _WORD_CHARS:
            self._word()
        else:
            self.position += 1
            self._note(char)

    def _word(self) -> None:
        start = self.position
        while self.position < len(self.code) and self.code[self.position] in _WORD_CHARS:
            self.position += 1
        self.previous = self.code[self.position - 1]
        self.previous_word = self.code[start : self.position]

    def _string(self, quote: str) -> None:
        line = self.line
        self.position += 1
        while self.position < len(self.code):
            char = self.code[self.position]
            if char == "\\":
                self.position += 2
            elif char == "\n":
                break
            elif char == quote:
                self.position += 1
                return
            else:
                self.position += 1
        raise _Failure(line, f"string opened with {quote} is never closed")

    def _close_bracket(self, char: str) -> None:
        frame = self.stack[-1]
        if len(self.stack) == 1 or frame.label != _CLOSERS[char]:
            raise _Failure(self.line, f"unexpected {char}")
        self.stack.pop()
        self.position += 1
        self._note(char)

    def _markup_can_start(self) -> bool:
        following = self._peek()
        if not (following.isalpha() or following in {">", "/"}):
            return False
        if self.previous_word:
            return self.previous_word in _EXPRESSION_KEYWORDS
        return not self.previous or self.previous in _EXPRESSION_PUNCTUATION

    def _open_markup(self) -> None:
        if self._peek() == "/":
            raise _Failure(self.line, "closing tag has no matching opening tag")
        start = self.position + 1
        end = self._name_end(start)
        rest = self.code[end:].lstrip()
        if rest.startswith(",") or rest.startswith("extends "):
            # A generic arrow function's type parameter: <T,>(x: T) => x
            self.position = end
            self._note("T")
            self.previous_word = self.code[start:end]
            return
        name = self.code[start:end]
        line = self.line
        self.position = end
        self.stack.append(_Frame(_Mode.TAG, f"<{name}>", line))

    def _name_end(self, start: int) -> int:
        end = start
        while end < len(self.code) and self.code[end] in _NAME_CHARS:
            end += 1
        return end

    def _tag(self) -> None:
        char = self.code[self.position]
        if self._startswith("/>"):
            self.stack.pop()
            self.position += 2
            self._note(")")
        elif char == ">":
            self.stack[-1].mode = _Mode.CHILDREN
            self.position += 1
        elif char == "{":
            self.stack.append(_Frame(_Mode.CODE, "{", self.line))
            self.position += 1
            self._note("{")
        elif char in {"'", '"'}:
            self._attribute_string(char)
        elif char.isspace() or char == "=" or char in _NAME_CHARS:
            self.position += 1
        else:
            raise _Failure(self.line, f"unexpected {char} inside {self.stack[-1].label}")

    def _attribute_string(self, quote: str) -> None:
        end = self.code.find(quote, self.position + 1)
        if end == -1:
            raise _Failure(self.line, "attribute value is never closed")
        self.position = end + 1

    def _children(self) -> None:
        char = self.code[self.position]
        if self._startswith("</"):
            self._closing_tag()
        elif char == "<":
            self._child_tag()
        elif char == "{":
            self.stack.append(_Frame(_Mode.CODE, "{", self.line))
            self.position += 1
            self._note("{")
        else:
            self.position += 1

    def _child_tag(self) -> None:
        start = self.position + 1
        end = self._name_end(start)
        self.stack.append(_Frame(_Mode.TAG, f"<{self.code[start:end]}>", self.line))
        self.position = end

    def _closing_tag(self) -> None:
        start = self.position + 2
        end = self._name_end(start)
        close = self.code.find(">", end)
        if close == -1 or self.code[end:close].strip():
            raise _Failure(self.line, "closing tag is malformed")
        expected = self.stack[-1].label
        found = f"<{self.code[start:end]}>"
        if expected != found:
            raise _Failure(self.line, f"{expected} is closed by </{found[1:]}")
        self.stack.pop()
        self.position = close + 1
        self._note(")")

    def _template(self) -> None:
        char = self.code[self.position]
        if char == "\\":
            self.position += 2
        elif char == "`":
            self.stack.pop()
            self.position += 1
            self._note("`")
        elif self._startswith("${"):
            self.stack.append(_Frame(_Mode.CODE, "{", self.line))
            self.position += 2
            self._note("{")
        else:
            self.position += 1


def find_tsx_issue(code: str) -> TsxIssue | None:
    """Return the first structural problem in the code, or ``None`` if it is balanced."""
    try:
        _Scanner(code).run()
    except _Failure as failure:
        return failure.issue
    return None
