# `MAKEFILES_DIR` __MUST__ be defined in the root Makefile
include $(MAKEFILES_DIR)/base.mk


HELP_MESSAGE += \
	\n$(TURQUOISE)$(BOLD)Quick Start:$(RESET) \
	\n  If you haven't already, create a virtual environment: $(GREEN)$(BOLD)make pyenv$(RESET) \
	\n  Then install the pre-compiled requirements: $(GREEN)$(BOLD)make install$(RESET)


VIRTUALENV_NAME := $(shell basename $(CURDIR))


.PHONY: pyenv
pyenv: # Create the Python virtual environment (pyenv).
	$(call ASSERT_SET,PYTHON_VERSION)
	$(call ASSERT_SET,VIRTUALENV_NAME)
	$(call INFO,Creating virtual environment)
	pyenv install -s $(PYTHON_VERSION)
	pyenv virtualenv $(PYTHON_VERSION) $(VIRTUALENV_NAME)
	pyenv local $(VIRTUALENV_NAME)
	python -m pip install --upgrade pip
