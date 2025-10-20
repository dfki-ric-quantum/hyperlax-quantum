
# Installation

[furo](https://github.com/pradyunsg/furo) is a theme based on sphinx.
```bash
pip install furo
```

```bash
mkdir docs
cd docs
sphinx-quickstart
```

Answer the questions. The most important ones are:
*   Separate source and build directories (y/n) [n] -> y
*   Project name: hyperlax
*   Author name(s): Your name/team
*   Project release []: 0.1.0
*   Project language [en]: en
*   Enable autodoc automatically (y/n) [n] -> y

Update the `html_theme` in `docs/source/conf.py`.

```python
html_theme = "furo"
```

# Building

Inside the `hyperlax/docs/` dir:
```bash
make html
```
or at the root `hyperlax` dir 

```bash
make doc
```

`
