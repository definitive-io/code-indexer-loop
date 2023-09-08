"""Code Splitter.

Implementation amalgamated from:
https://docs.sweep.dev/blogs/chunking-improvements
https://docs.sweep.dev/blogs/chunking-2m-files
https://github.com/jerryjliu/llama_index/pull/7100

"""

import re
from dataclasses import dataclass
from typing import List, Optional, Union

import tiktoken
from tree_sitter import Node


class MaxChunkLengthExceededError(Exception):
    pass


@dataclass
class Span:
    # Represents a slice of a string
    start: int = 0
    end: int = 0

    def __post_init__(self):
        # If end is None, set it to start
        if self.end is None:
            self.end = self.start

    def extract(self, s: bytes) -> bytes:
        # Grab the corresponding substring of string s by bytes
        return s[self.start : self.end]

    def extract_lines(self, s: str) -> str:
        lines = s.split("\n")
        selected_lines = lines[self.start : self.end]
        joined = "\n".join(selected_lines)
        # if selection doesn't extend to the last line, add the missing newline
        if self.end < len(lines):
            joined += "\n"
        return joined

    def __add__(self, other: Union["Span", int]) -> "Span":
        # e.g. Span(1, 2) + Span(2, 4) = Span(1, 4) (concatenation)
        # There are no safety checks: Span(a, b) + Span(c, d) = Span(a, d)
        # and there are no requirements for b = c.
        if isinstance(other, int):
            return Span(self.start + other, self.end + other)
        elif isinstance(other, Span):
            return Span(self.start, other.end)
        else:
            raise NotImplementedError()

    def __len__(self) -> int:
        # i.e. Span(a, b) = b - a
        return self.end - self.start


class TokenCounter:
    default_model: str
    initialized_models = {}

    def __init__(self, default_model: str):
        self.default_model = default_model

    def count(self, text: str, model: Optional[str] = None):
        if model is None:
            model = self.default_model

        if model not in self.initialized_models:
            try:
                self.initialized_models[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                raise KeyError(f"Model {model} not supported.")

        return len(self.initialized_models[model].encode(text, disallowed_special=()))

    def count_chunk(self, chunk: Span, source_code: bytes, model: Optional[str] = None):
        return self.count(chunk.extract(source_code).decode("utf-8"), model)


class CodeSplitter:
    """Split code using a AST parser."""

    language: str
    target_chunk_tokens: int
    max_chunk_tokens: int
    enforce_max_chunk_tokens: bool
    coalesce: int
    token_counter: TokenCounter

    def __init__(
        self,
        language: str,
        target_chunk_tokens: int,
        max_chunk_tokens: int,
        enforce_max_chunk_tokens: bool,
        coalesce: int,
        token_model: str,
    ):
        self.token_counter = TokenCounter(default_model=token_model)
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.enforce_max_chunk_tokens = enforce_max_chunk_tokens
        self.language = language
        self.coalesce = coalesce

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "CodeSplitter"

    def chunk_tree(
        self,
        tree,
        source_code: bytes,
    ) -> list[Span]:
        # 1. Recursively form chunks
        def chunk_node(node: Node) -> list[Span]:
            chunks: list[Span] = []
            current_chunk: Span = Span(node.start_byte, node.start_byte)
            node_children = node.children
            for child in node_children:
                child_token_len = self.token_counter.count_chunk(Span(child.start_byte, child.end_byte), source_code)
                child_and_current_token_len = self.token_counter.count_chunk(
                    Span(child.start_byte, child.end_byte), source_code
                ) + self.token_counter.count_chunk(current_chunk, source_code)

                if child_token_len > self.target_chunk_tokens:
                    if child_token_len > self.max_chunk_tokens and self.enforce_max_chunk_tokens:
                        raise MaxChunkLengthExceededError(
                            f"Chunk token length {child_token_len} exceeds maximum {self.max_chunk_tokens}."
                        )

                    chunks.append(current_chunk)
                    current_chunk = Span(child.end_byte, child.end_byte)
                    chunks.extend(chunk_node(child))
                elif child_and_current_token_len > self.target_chunk_tokens:
                    if child_and_current_token_len > self.max_chunk_tokens and self.enforce_max_chunk_tokens:
                        raise MaxChunkLengthExceededError(
                            f"Chunk token length {child_and_current_token_len}"
                            f" exceeds maximum {self.max_chunk_tokens}."
                        )
                    chunks.append(current_chunk)
                    current_chunk = Span(child.start_byte, child.end_byte)
                else:
                    current_chunk += Span(child.start_byte, child.end_byte)

            final_chunk_token_len = self.token_counter.count_chunk(current_chunk, source_code)
            if final_chunk_token_len > self.max_chunk_tokens and self.enforce_max_chunk_tokens:
                raise MaxChunkLengthExceededError(
                    f"Chunk token length {final_chunk_token_len} exceeds maximum {self.max_chunk_tokens}."
                )
            chunks.append(current_chunk)
            return chunks

        chunks = chunk_node(tree.root_node)

        # Filter empty chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 0]

        # Early return if there is no chunk
        if len(chunks) == 0:
            return []
        # Early return if there is only one chunk
        if len(chunks) < 2:
            return [Span(0, len(chunks[0]))]

        # Filling in the gaps
        # by aligning end of one chunk with start of next
        chunks[0].start = 0
        for prev, curr in zip(chunks[:-1], chunks[1:]):
            prev.end = curr.start
        curr.end = len(source_code)

        # Combining small chunks with bigger ones
        new_chunks = []
        aggregated_chunk = Span(0, 0)
        aggregated_chunk_token_len = 0
        for chunk in chunks:
            # Check if the combined chunk exceeds target_chunk_tokens
            # Note, at this point no chunk exceeds max_chunk_tokens
            # if max_chunk_tokens is enforced.
            chunk_token_len = self.token_counter.count_chunk(chunk, source_code)
            if chunk_token_len > self.target_chunk_tokens:
                new_chunks.append(aggregated_chunk)
                new_chunks.append(chunk)
                aggregated_chunk = Span(chunk.end, chunk.end)
                aggregated_chunk_token_len = 0
            elif aggregated_chunk_token_len + chunk_token_len > self.target_chunk_tokens:
                new_chunks.append(aggregated_chunk)
                aggregated_chunk = Span(chunk.start, chunk.end)
                aggregated_chunk_token_len = chunk_token_len
            else:
                # Combined chunk does not exceed target_chunk_tokens
                # so we add the current chunk to the aggregated_chunk.
                # Note, there is no need to check whether the combined chunk
                # exceeds max_chunk_tokens because we have already checked.
                aggregated_chunk += chunk
                aggregated_chunk_token_len += chunk_token_len
                if aggregated_chunk_token_len > self.coalesce:
                    new_chunks.append(aggregated_chunk)
                    aggregated_chunk = Span(chunk.end, chunk.end)
                    aggregated_chunk_token_len = 0

        if len(aggregated_chunk) > 0:
            new_chunks.append(aggregated_chunk)

        # Changing line numbers
        line_chunks = [
            Span(
                self.get_line_number(chunk.start, source_code),
                self.get_line_number(chunk.end, source_code),
            )
            for chunk in new_chunks
        ]

        # Eliminating empty chunks
        line_chunks = [chunk for chunk in line_chunks if len(chunk) > 0]
        return line_chunks

    def split_and_keep_newline(self, byte_str):
        return re.split(b"(?<=\n)", byte_str)

    def get_line_number(self, index: int, source_code: bytes) -> int:
        total_chars = 0
        for line_number, line in enumerate(self.split_and_keep_newline(source_code), start=1):
            total_chars += len(line)
            if total_chars > index:
                return line_number - 1
        return line_number

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST."""
        try:
            import tree_sitter_languages
        except ImportError:
            raise ImportError("Please install tree_sitter_languages to use CodeSplitter.")

        try:
            parser = tree_sitter_languages.get_parser(self.language)
        except Exception as e:
            print(
                f"Could not get parser for language {self.language}. Check "
                "https://github.com/grantjenks/py-tree-sitter-languages#license "
                "for a list of valid languages."
            )
            raise e

        tree = parser.parse(text.encode("utf-8"))
        if not tree.root_node.children or tree.root_node.children[0].type != "ERROR":
            line_spans = self.chunk_tree(tree, text.encode("utf-8"))
            chunks = [line_span.extract_lines(text) for line_span in line_spans]
            return chunks
        else:
            raise ValueError(f"Could not parse code with language {self.language}.")
