import os

import pytest

from cil.code_splitter import (CodeSplitter, MaxChunkLengthExceededError,
                               TokenCounter)

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def create_code_splitter(language="python", target_chunk_tokens=5, max_chunk_tokens=200, enforce_max_chunk_tokens=True):
    return CodeSplitter(
        language=language,
        target_chunk_tokens=target_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        enforce_max_chunk_tokens=enforce_max_chunk_tokens,
        token_model="gpt-4",
        coalesce=50,
    )


def test_code_splitter():
    python_code_splitter = create_code_splitter()
    chunks = python_code_splitter.split_text(
        """def foo():
    print("Hello, world!")

print(1)"""
    )
    assert chunks[0].startswith("def foo():")
    assert not chunks[0].endswith('")')


def test_code_splitter_newlines():
    python_code_splitter = create_code_splitter()
    chunks = python_code_splitter.split_text(
        """
def foo():
    print("Hello, world!")

print(1)

"""
    )
    assert chunks[0].startswith("\ndef foo():")
    assert not chunks[0].endswith('")')
    assert chunks[-1].endswith("\n\n")


def test_code_splitter_raise():
    python_code_splitter = create_code_splitter(max_chunk_tokens=5)
    with pytest.raises(MaxChunkLengthExceededError):
        python_code_splitter.split_text(
            """
def mostdefinitelynotlessthan5tokens():
    pass
"""
        )


def test_code_splitter_noraise():
    python_code_splitter = create_code_splitter(max_chunk_tokens=5, enforce_max_chunk_tokens=False)
    python_code_splitter.split_text(
        """
def mostdefinitelynotlessthan5tokens():
    pass
"""
    )


def test_code_splitter_token_lengths():
    tc = TokenCounter(default_model="gpt-4")
    max_chunk_tokens = 20
    python_code_splitter = create_code_splitter(
        max_chunk_tokens=max_chunk_tokens, target_chunk_tokens=max_chunk_tokens // 2
    )
    source_code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

add(1, 2)
"""
    chunks = python_code_splitter.split_text(source_code)
    joined_chunks = "".join(chunks)
    assert source_code == joined_chunks

    chunk_lengths = [tc.count(chunk) for chunk in chunks]
    assert all([chunk_length <= max_chunk_tokens for chunk_length in chunk_lengths])


def test_long_file():
    hard_file_path = os.path.join(THIS_FILE_DIR, "test_api_dummy_file.py.txt")
    with open(hard_file_path, "r") as f:
        source_code = f.read()

    python_code_splitter = create_code_splitter(target_chunk_tokens=1000, max_chunk_tokens=9000)
    chunks = python_code_splitter.split_text(source_code)
    joined_chunks = "".join(chunks)
    assert source_code == joined_chunks


def test_sql():
    sql_file_path = os.path.join(THIS_FILE_DIR, "test_api_dummy_sql.sql.txt")
    with open(sql_file_path, "r") as f:
        source_code = f.read()

    sql_code_splitter = CodeSplitter(
        language="sql",
        target_chunk_tokens=10,
        max_chunk_tokens=1000,
        enforce_max_chunk_tokens=True,
        token_model="gpt-4",
        coalesce=50,
    )

    chunks = sql_code_splitter.split_text(source_code)
    joined_chunks = "".join(chunks)
    assert source_code == joined_chunks
