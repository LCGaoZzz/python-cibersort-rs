# Contributing

Bug reports and focused pull requests are welcome.

1. Create a branch from `main`.
2. Install the development environment with `maturin develop --release`.
3. Run `pytest -m "not slow"` while iterating.
4. Run the complete `pytest` suite before opening a pull request that changes
   numerical behavior.
5. Include an oracle or focused regression test for numerical changes.

Do not commit the original CIBERSORT script, LM22, private datasets, generated
build directories, or benchmark output containing local paths. Performance
claims should include raw rows, hardware details, package versions, and a
description of background load.
