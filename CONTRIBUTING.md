Thank you for your interest in contributing to TorchPipe!

This document (CONTRIBUTING.md) covers some technical
aspects of contributing to TorchPipe.

Please feel free to raise an issue, a merge request, or a commit.

# Table of Contents

<!-- toc -->

- [Developing TorchPipe](#developing-torchpipe)
- [Codebase structure](#codebase-structure)
- [Unit testing](#unit-testing)
  - [Python Unit Testing](#python-unit-testing)
- [Writing documentation](#writing-documentation)

<!-- tocstop -->

## Developing TorchPipe
Follow the instructions for [installing TorchPipe from source](https://torchpipe.github.io/docs/installation). 

The way to modify the code:

- Submit a merge request to the develop branch.

Special requirements for C++：
- All code needs to be exception-safe.
- Manual program termination is not allowed, but instead replace it with throwing exceptions.

## Codebase structure

* [examples](./examples/) - examples
* [torchpipe](./torchpipe) - main source
  * [csrc](torchpipe/csrc) - C++ core library for TorchPipe 
    * [python](torchpipe/csrc/python)
    * [opencv](torchpipe/csrc/opencv)
    * [schedule](torchpipe/csrc/schedule)
    * [pipeline](torchpipe/csrc/pipeline)
  * [utils](torchpipe/utils) 
* [test](test) - Python unit tests for TorchPipe Python frontend.
* [.clang-format](.clang-format)

## Unit testing

### Python Unit Testing

```bash
cd test
pip install -r requirements.txt 
pytest .
```

## Writing documentation
It is welcome to write documentation.