# Code Indexer Loop

[![PyPI version](https://badge.fury.io/py/code-indexer-loop.svg?v=2)](https://pypi.org/project/code-indexer-loop/)
[![License](https://img.shields.io/github/license/definitive-io/code-indexer-loop?v=2)](LICENSE)
[![Forks](https://img.shields.io/github/forks/definitive-io/code-indexer-loop?v=2)](https://github.com/definitive-io/code-indexer-loop/network)
[![Stars](https://img.shields.io/github/stars/definitive-io/code-indexer-loop?v=2)](https://github.com/definitive-io/code-indexer-loop/stargazers)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com?style=social&label=Follow%20%40DefinitiveIO)](https://twitter.com/definitiveio)
[![Discord](https://dcbadge.vercel.app/api/server/CPJJfq87Vx?compact=true&style=flat)](https://discord.gg/CPJJfq87Vx)


**Code Indexer Loop** is a Python library designed to index and retrieve code snippets. 

It uses the useful indexing utilities of the **LlamaIndex** library and the multi-language **tree-sitter** library to parse the code from many popular programming languages. **tiktoken** is used to right-size retrieval based on number of tokens and **LangChain** is used to obtain embeddings (defaults to **OpenAI**'s `text-embedding-ada-002`) and store them in an embedded **ChromaDB** vector database. **watchdog** is used for continuous updating of the index based on file system events.

Read the [launch blog post](https://www.definitive.io/blog/open-sourcing-code-indexer-loop) for more details about why we've built this!

## Installation:
Use `pip` to install Code Indexer Loop from PyPI.
```
pip install code-indexer-loop
```

## Usage:
1. Import necessary modules:
```python
from code_indexer_loop.api import CodeIndexer
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

## Examples
Check out the [basic_usage](examples/basic_usage.ipynb) notebook for a quick overview of the API.

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

## tree-sitter
Using `tree-sitter` for parsing, the chunks are broken only at valid node-level string positions in the source file. This avoids breaking up e.g. function and class definitions.

### Supported languages:
C, C++, C#, Go, Haskell, Java, Julia, JavaScript, PHP, Python, Ruby, Rust, Scala, Swift, SQL, TypeScript

Note, we're mainly testing Python support. Use other languages at your own peril.

## Contributing
Pull requests are welcome. Please make sure to update tests as appropriate. Use tools provided within `dev` dependencies to maintain the code standard.

### Tests
Run the unit tests by invoking `pytest` in the root.

## License
Please see the LICENSE file provided with the source code.

## Attribution
We'd like to thank the Sweep AI for publishing their ideas about code chunking. Read their blog posts about the topic [here](https://docs.sweep.dev/blogs/chunking-2m-files) and [here](https://docs.sweep.dev/blogs/chunking-improvements). The implementation in `code_indexer_loop` is modified from their original implementation mainly to limit based on tokens instead of characters and to achieve perfect document reconstruction (`"".join(chunks) == original_source_code`).
