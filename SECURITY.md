# Security policy

## Supported versions

Security fixes are made for the latest release of Strop on PyPI. Please upgrade before you
report a problem.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting
instead:

1. Open the [Security tab](https://github.com/baranylcn/strop/security) of the repository.
2. Choose **Report a vulnerability**.
3. Describe what you found, the Strop version, and the steps to reproduce it.

Strop is maintained on a best-effort basis. You will get a reply as soon as possible, and a fix
will be released with a changelog entry once the problem is confirmed.

## What counts as a security issue

Strop runs on your machine and sends your dataset to the model providers you configure. These
are examples of what to report:

- API keys or other secrets that end up in logs, error messages, the database, or exported
  reports.
- Dataset rows reaching a model, a file, or a provider that the configuration does not name.
- Rows from the test partition reaching the optimizer or analyzer.
- Code from a dataset being executed, or `metric.py` running outside its separate process.
- A way to write outside the project directory through a command-line option or a file name.

Problems in a model provider, in LiteLLM, or in another dependency should be reported to that
project. A `metric.py` file is code you wrote and runs as written, so what it does is not a Strop
vulnerability.
