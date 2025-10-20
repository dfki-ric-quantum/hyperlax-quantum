.PHONY: doc

doc:
	@echo "Building Sphinx HTML documentation..."
	@$(MAKE) -C docs html

clean-doc:
	@echo "Cleaning Sphinx build directory..."
	@$(MAKE) -C docs clean
