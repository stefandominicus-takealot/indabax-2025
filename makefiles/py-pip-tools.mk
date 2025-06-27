# `MAKEFILES_DIR` __MUST__ be defined in the root Makefile
include $(MAKEFILES_DIR)/base.mk


.PHONY: pip-tools
pip-tools:
	@python -m pip install pip-tools


%.txt: %.in | pip-tools
	$(call INFO,Compiling $*.txt)
	CUSTOM_COMPILE_COMMAND="make update" \
	pip-compile --verbose --upgrade --strip-extras --no-emit-index-url --resolver=backtracking $<


.PHONY: requirements
requirements: requirements.in # Compile requirements (pip-compile).
	@touch requirements.in
	@$(MAKE) requirements.txt


.PHONY: sync
sync: requirements.txt | pip-tools # Sync dependencies (pip-sync).
	$(call INFO,Syncing dependencies)
	pip-sync requirements.txt


.PHONY: update
update: requirements sync # Compile requirements & sync dependencies.


.PHONY: install
install: requirements.txt # Install dependencies (pip).
	$(call INFO,Installing dependencies)
	python -m pip install -r requirements.txt
