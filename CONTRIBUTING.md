# Contributing to hyperlax-quantum

Please inform the maintainer as early as possible about your planned feature developments, extensions, or bugfixes that you are working on. An easy way is to open an issue or a pull request in which you explain what you are trying to do.

## Pull Requests

The preferred way to contribute to hyperlax is to fork the [hyperlax-quantum](https://github.com/dfki-ric-quantum/hyperlax-quantum) on GitHub, then submit a "pull request" (PR):

1. [Create an account](https://github.com/signup/free) on GitHub if you do not already have one.

2. Fork the [hyperlax-quantum repository](https://github.com/dfki-ric-quantum/hyperlax-quantum): click on the 'Fork' button near the top of the page. This creates a copy of the code under your account on the GitHub server.

3. Clone this copy to your local disk:

        $ git clone git@github.com:dfki-ric-quantum/hyperlax-quantum.git

4. Create a branch to hold your changes:

        $ git checkout -b my-feature

    and start making changes. Never work in the ``master`` branch!

5. Work on this copy, on your computer, using Git to do the version control. When you're done editing, do::

        $ git add modified_files
        $ git commit

    to record your changes in Git, then push them to GitHub with::

       $ git push -u origin my-feature

Finally, go to the web page of the your fork of the bolero repo, and click 'Pull request' to send your changes to the maintainers for review. request.

## Merge Policy

Direct pushes to the `main` branch of **hyperlax-quantum** are generally not allowed.
The only exceptions are:

* **Tiny changes** — spelling fixes, small documentation updates, or trivial refactorings.
* **Urgent bugfixes** — fixes required to restore build stability or address critical production issues.
* **Maintenance commits** — dependency updates or CI configuration adjustments that do not affect functionality.

A "tiny change" must **not** introduce new features, must **preserve backwards compatibility**, and must pass **all tests** before being pushed. These direct changes may only be made by the **maintainer**.

All other contributions must be submitted as **pull requests**:

1. **Create a branch** for your work.
2. **Open a pull request** against the `main` branch.
3. Your PR must be **reviewed by at least one other developer** before merging.
4. PRs are merged by the **maintainer** once they meet all requirements.

**Additional rules:**

* New features must include both **documentation** and **tests**.
* Breaking changes must be **discussed in advance**, and should include **deprecation warnings** before removal.
* All PRs must pass **CI checks** before merging.

## Funding

This work was funded by the German Ministry of Economic Affairs and Climate Action (BMWK) and the German Aerospace Center (DLR) in project QuBER-KI (grants: 50RA2207A, 50RA2207B).
