# `MAKEFILES_DIR` __MUST__ be defined in the root Makefile
include $(MAKEFILES_DIR)/base.mk


.PHONY: help
help: # Show this help message.
	@echo "Usage: $(YELLOW)$(BOLD)make [TARGET]$(RESET)\n"
	@echo "$(YELLOW)$(BOLD)TARGETS$(RESET)"
	@grep -Eh '^[a-zA-Z_%\.-]+:.*?# .*$$' $(MAKEFILE_LIST) \
		| tac \
		| awk 'BEGIN {FS = ":.*?# "}; \
		{ \
			if ( !seen[$$1]++ ) \
			{ \
				printf " $(BOLD)>$(RESET) $(TURQUOISE)%-24s$(RESET)%s\n", $$1, $$2 \
			} \
		}' \
		| sort
	@echo "$(HELP_MESSAGE)"


.DEFAULT_GOAL = help
