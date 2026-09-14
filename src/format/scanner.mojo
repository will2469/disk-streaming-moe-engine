# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pemindai JSON byte-level untuk subset header safetensors (M0-W1)."""

from format.types import STError
from format.utf8 import parse_json_escape
from std.collections import List


struct Scanner(Movable):
    """Pemindai JSON byte-level untuk subset header safetensors."""

    var buf: List[UInt8]
    var pos: Int
    var shard: String

    def __init__(out self, var buf: List[UInt8], shard: String):
        self.buf = buf^
        self.pos = 0
        self.shard = shard

    def eof(self) -> Bool:
        return self.pos >= len(self.buf)

    def peek(self) -> Int:
        return Int(self.buf[self.pos])

    def skip_ws(mut self):
        while not self.eof():
            var b = self.peek()
            if b == 32 or b == 10 or b == 13 or b == 9:
                self.pos += 1
            else:
                break

    def expect(mut self, want: Int) raises:
        self.skip_ws()
        if self.eof() or self.peek() != want:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        String("expected byte ", want, " at ", self.pos),
                        self.shard,
                        "",
                    )
                )
            )
        self.pos += 1

    def parse_uint(mut self) raises -> Int:
        self.skip_ws()
        var v = 0
        var ndigits = 0
        while not self.eof():
            var b = self.peek()
            if b < 48 or b > 57:
                break
            if v > 900719925474099:
                raise Error(
                    String(
                        STError(
                            "INVALID_HEADER", "integer overflow", self.shard, ""
                        )
                    )
                )
            v = v * 10 + (b - 48)
            ndigits += 1
            self.pos += 1
        if ndigits == 0:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        String("expected integer at ", self.pos),
                        self.shard,
                        "",
                    )
                )
            )
        return v

    def parse_string(mut self) raises -> String:
        self.expect(34)
        var out = List[UInt8]()
        while True:
            if self.eof():
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "unterminated string",
                            self.shard,
                            "",
                        )
                    )
                )
            var b = self.peek()
            if b == 34:
                self.pos += 1
                break
            if b == 92:
                self.pos += 1
                parse_json_escape(out, self.buf, self.pos, self.shard)
            elif b < 32:
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            String("raw control < 0x20 at ", self.pos),
                            self.shard,
                            "",
                        )
                    )
                )
            else:
                out.append(UInt8(b))
                self.pos += 1
        return String(from_utf8_lossy=Span(out))

    def skip_value(mut self) raises:
        self.skip_ws()
        if self.eof():
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR", "unexpected eof", self.shard, ""
                    )
                )
            )
        var b = self.peek()
        if b == 34:
            _ = self.parse_string()
            return
        if (b >= 48 and b <= 57) or b == 45:
            if b == 45:
                self.pos += 1
            _ = self.parse_uint()
            self.skip_ws()
            if not self.eof():
                var c = self.peek()
                if c == 46 or c == 69 or c == 101:
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "pecahan/eksponen di luar subset",
                                self.shard,
                                "",
                            )
                        )
                    )
            return
        if b == 116:
            self.expect_literal("true")
            return
        if b == 102:
            self.expect_literal("false")
            return
        if b == 110:
            self.expect_literal("null")
            return
        if b != 123 and b != 91:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        String("bad value at ", self.pos),
                        self.shard,
                        "",
                    )
                )
            )
        var stack = List[Int]()
        stack.append(b)
        self.pos += 1
        while len(stack) > 0:
            if self.eof():
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "unterminated composite",
                            self.shard,
                            "",
                        )
                    )
                )
            var c = self.peek()
            if c == 34:
                _ = self.parse_string()
            elif c == 123 or c == 91:
                stack.append(c)
                self.pos += 1
            elif c == 125 or c == 93:
                var want = 125
                if stack[len(stack) - 1] == 91:
                    want = 93
                if c != want:
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                String("tutup silang at ", self.pos),
                                self.shard,
                                "",
                            )
                        )
                    )
                _ = stack.pop()
                self.pos += 1
            elif c == 44 or c == 58:
                self.pos += 1
            elif (c >= 48 and c <= 57) or c == 45:
                if c == 45:
                    self.pos += 1
                _ = self.parse_uint()
                self.skip_ws()
                if not self.eof():
                    var d = self.peek()
                    if d == 46 or d == 69 or d == 101:
                        raise Error(
                            String(
                                STError(
                                    "JSON_PARSE_ERROR",
                                    "pecahan/eksponen di luar subset",
                                    self.shard,
                                    "",
                                )
                            )
                        )
            elif c == 32 or c == 10 or c == 13 or c == 9:
                self.pos += 1
            else:
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            String("byte liar at ", self.pos),
                            self.shard,
                            "",
                        )
                    )
                )

    def expect_literal(mut self, word: String) raises:
        var wb = word.as_bytes()
        for i in range(len(wb)):
            if self.eof() or self.peek() != Int(wb[i]):
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            String("bad literal at ", self.pos),
                            self.shard,
                            "",
                        )
                    )
                )
            self.pos += 1

    def parse_int_array(mut self) raises -> List[Int]:
        var out = List[Int]()
        self.expect(91)
        self.skip_ws()
        if not self.eof() and self.peek() == 93:
            self.pos += 1
            return out^
        while True:
            var v = self.parse_uint()
            out.append(v)
            self.skip_ws()
            if self.eof():
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "unterminated array",
                            self.shard,
                            "",
                        )
                    )
                )
            var b = self.peek()
            self.pos += 1
            if b == 93:
                break
            if b != 44:
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "expected , or ]",
                            self.shard,
                            "",
                        )
                    )
                )
        return out^
