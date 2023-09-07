# CIL: Code Indexer Loop

[![PyPI version](https://badge.fury.io/py/cil.svg)](https://pypi.org/project/cil/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2-yellow.svg)](https://opensource.org/license/apache-2-0/)
[![Forks](https://img.shields.io/github/forks/definitive-io/cil)](https://github.com/definitive-io/cil/network)
[![Stars](https://img.shields.io/github/stars/definitive-io/cil)](https://github.com/definitive-io/cil/stargazers)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com?style=social&label=Follow%20%40DefinitiveIO)](https://twitter.com/definitiveio)
[![Discord](https://dcbadge.vercel.app/api/server/CPJJfq87Vx?compact=true&style=flat)](https://discord.gg/CPJJfq87Vx)


**CIL** is a Python library designed to index and retrieve code snippets. 

It uses the useful indexing utilities of the **LlamaIndex** library and the multi-language **tree-sitter** library to parse the code from many popular programming languages. **tiktoken** is used to count tokens in a code snippet and **LangChain** to obtain embeddings (defaults to **OpenAI**'s `text-embedding-ada-002`) and store them in an embedded **ChromaDB** vector database. It uses **watchdog** for continuous updating of the index based on file system events.

## Installation:
Use `pip` to install Code Indexer Loop from PyPI.
```
pip install cil
```

## Usage:
1. Import necessary modules:
```python
from cil.api import CodeIndexer
```
2. Create a CodeIndexer object and have it watch for changes:
```python
indexer = CodeIndexer(src_dir="path/to/code/", watch=True)
```
3. Use `.query` to perform a search query:
```python
query = "pandas"
print(indexer.query(query)[0:30])
```

You can also use `indexer.query_nodes` to get the nodes of a query or `indexer.query_documents` to receive the entire source code files.

Note that if you edit any of the source code files in the `src_dir` it will efficiently re-index those files using `watchdog` and an `md5` based caching mechanism. This results in up-to-date embeddings every time you query the index.

## tree-sitter
Using `tree-sitter` for parsing, the chunks are broken only at valid node-level string positions in the source file. This avoids breaking up e.g. function and class definitions.

## Token limits
You can configure token limits for the chunks through the CodeIndexer constructor:

```python
indexer = CodeIndexer(
    src_dir="path/to/code/", watch=True,
    target_chunk_tokens = 300,
    max_chunk_tokens = 1000,
    enforce_max_chunk_tokens = False,
    coalesce = 50
    token_model = "gpt-4"
)
```

Note you can choose whether the `max_chunk_tokens` is enforced. If it is, it will raise an exception in case there is no semantic parsing that respects the `max_chunk_tokens`.

The `coalesce` argument controls the limit of combining smaller chunks into single chunks to avoid having many very small chunks. The unit for `coalesce` is also tokens.

## Supported languages:
C, C++, C#, Go, Haskell, Java, Julia, JavaScript, PHP, Python, Ruby, Rust, Scala, Swift, SQL, TypeScript

Note, we're mainly testing Python support. Use other languages at your own peril.

## Contributing
Pull requests are welcome. Please make sure to update tests as appropriate. Use tools provided within `dev` dependencies to maintain the code standard.

### Tests
Run the unit tests by invoking `pytest` in the root.

## Examples
Check out the [basic_usage](examples/basic_usage.ipynb) notebook for a quick overview of the API.

## License
Please see the LICENSE file provided with the source code.

## Attribution
We'd like to thank the Sweep AI for publishing their ideas about code chunking. Read their blog posts about the topic [here](https://docs.sweep.dev/blogs/chunking-2m-files) and [here](https://docs.sweep.dev/blogs/chunking-improvements). The implementation in `cil` is modified from their original implementation mainly to limit based on tokens instead of characters and to achieve perfect document reconstruction (`"".join(chunks) == original_source_code`).