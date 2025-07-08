MAKEFILES_DIR = makefiles


include $(MAKEFILES_DIR)/help.mk
include $(MAKEFILES_DIR)/py-pip-tools.mk


# TFX doesn't support newer Python versions yet.
# See https://pypi.org/project/tfx/
PYTHON_VERSION = 3.10
include $(MAKEFILES_DIR)/py-env.mk
