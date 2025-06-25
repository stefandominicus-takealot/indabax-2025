# Colours
BOLD      := $(shell tput -Txterm bold)
RED       := $(shell tput -Txterm setaf 1)
GREEN     := $(shell tput -Txterm setaf 2)
YELLOW    := $(shell tput -Txterm setaf 3)
BLUE      := $(shell tput -Txterm setaf 4)
PURPLE    := $(shell tput -Txterm setaf 5)
TURQUOISE := $(shell tput -Txterm setaf 6)
WHITE     := $(shell tput -Txterm setaf 7)
RESET     := $(shell tput -Txterm sgr0)


# Pretty Printing
INFO = @echo "$(TURQUOISE)$(BOLD)$(1)$(RESET)" # $(info ...) runs at read-time
WARNING = @echo "$(YELLOW)$(BOLD)$(1)$(RESET)" # $(warning ...) runs at read-time
ERROR = @echo "$(RED)$(BOLD)$(1)$(RESET)" && exit 1 # $(error ...) runs at read-time


# Booleans
TRUE = true
FALSE = false


# Assertions
# Ensure a variable's value is either true or false
# Example use: $(call ASSERT_BOOL,my_bool)
ASSERT_BOOL = $(if $(filter $($(1)),$(TRUE) $(FALSE)),,$(call ERROR,$(1)=$($(1)) is neither $(TRUE) nor $(FALSE)))
# Ensure a variable is not undefined (empty OK)
# Example use: $(call ASSERT_DEFINED,MY_ENV)
ASSERT_DEFINED = $(if $(filter undefined,$(origin $(1))),$(call ERROR,$(1) must be defined))
# Ensure a variable is set (non-empty)
# Example use: $(call ASSERT_SET,MY_ENV)
ASSERT_SET = $(if $($(1)),,$(call ERROR,$(1) must be set))


# Character References
COMMA := ,
