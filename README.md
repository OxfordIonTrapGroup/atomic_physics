# Atomic Physics

## Setup
Pre-requisites:

* [Poetry ^1.1](https://python-poetry.org/)
    * `pip3 install poetry`
    * [Poe the poet task runner](https://github.com/nat-n/poethepoet)
        * `pip3 install poethepoet`

        To install dependencies and create virtual env: `poetry install`.

## Testing
* `poe test` can runs tests
* `poe flake` runs flake8.
* tests and flake must pass before PRs can be merged (master is branch protected)

## Automatic formatting
`poe fmt` will use `black` to automatically format the code
