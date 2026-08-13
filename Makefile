# Chris Llama / Chris-GPT build
#
# BACKEND=auto : use AdaptiveCpp when `acpp` is available, otherwise use the
#                serial CPU reference backend.
# BACKEND=acpp : require AdaptiveCpp and enable the SYCL backend.
# BACKEND=cpu  : build the serial CPU reference backend.

CC ?= gcc
ACPP ?= acpp
BACKEND ?= auto
NATIVE ?= 0
ACPP_TARGETS ?= generic

BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
TEST_BUILD_DIR := $(BUILD_DIR)/tests

CFLAGS := -std=c11 -O3 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wpedantic
LDLIBS := -lm

ifeq ($(NATIVE),1)
CFLAGS += -march=native
CXXFLAGS += -march=native
endif

ifeq ($(BACKEND),auto)
  ifneq ($(shell command -v $(ACPP) 2>/dev/null),)
    BACKEND := acpp
  else
    BACKEND := cpu
  endif
endif

ifeq ($(BACKEND),acpp)
  ifeq ($(shell command -v $(ACPP) 2>/dev/null),)
    $(error BACKEND=acpp requested, but '$(ACPP)' was not found in PATH)
  endif
  BACKEND_OBJ := $(OBJ_DIR)/backend_acpp.o
  LINKER := $(ACPP)
  LINKFLAGS := --acpp-targets="$(ACPP_TARGETS)"
  BACKEND_DESC := AdaptiveCpp/SYCL ($(ACPP_TARGETS))
else ifeq ($(BACKEND),cpu)
  BACKEND_OBJ := $(OBJ_DIR)/backend_cpu.o
  LINKER := $(CC)
  LINKFLAGS :=
  BACKEND_DESC := CPU reference
else
  $(error BACKEND must be auto, acpp, or cpu)
endif

COMMON_SRC := main.c gguf.c quant.c tokenizer.c gpt2.c sampler.c
COMMON_OBJ := $(COMMON_SRC:%.c=$(OBJ_DIR)/%.o)
OBJ := $(COMMON_OBJ) $(BACKEND_OBJ)
BIN := $(BIN_DIR)/chris_llama

.PHONY: all clean test info dirs

all: info $(BIN)

info:
	@echo "Building backend: $(BACKEND_DESC)"
	@echo "Build directory: $(BUILD_DIR)"

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) $(TEST_BUILD_DIR)

$(BIN): $(OBJ) | dirs
	$(LINKER) $(LINKFLAGS) $(OBJ) -o $@ $(LDLIBS)

$(OBJ_DIR)/%.o: %.c | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/backend_acpp.o: backend_acpp.cpp backend.h | dirs
	$(ACPP) --acpp-targets="$(ACPP_TARGETS)" $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/backend_cpu.o: backend_cpu.c backend.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

# The same functional tests are used for both backends. With BACKEND=acpp the
# backend test also launches the SYCL MatVec kernel on the selected device.
test: $(BIN) | dirs
	$(CC) $(CFLAGS) tests/test_quant.c quant.c -o $(TEST_BUILD_DIR)/test_quant -lm
	$(TEST_BUILD_DIR)/test_quant
	$(CC) $(CFLAGS) -c tests/test_backend.c -o $(TEST_BUILD_DIR)/test_backend.o
ifeq ($(BACKEND),acpp)
	$(LINKER) $(LINKFLAGS) $(TEST_BUILD_DIR)/test_backend.o $(BACKEND_OBJ) -o $(TEST_BUILD_DIR)/test_backend -lm
else
	$(CC) $(TEST_BUILD_DIR)/test_backend.o $(BACKEND_OBJ) -o $(TEST_BUILD_DIR)/test_backend -lm
endif
	$(TEST_BUILD_DIR)/test_backend
	python3 tests/generate_tiny_gguf.py $(TEST_BUILD_DIR)/tiny-gpt2.gguf
	$(BIN) $(TEST_BUILD_DIR)/tiny-gpt2.gguf --inspect >/dev/null
	$(BIN) $(TEST_BUILD_DIR)/tiny-gpt2.gguf --tokenize "Hi" | grep -q "tokens\[2\] = 72 105"
	$(BIN) $(TEST_BUILD_DIR)/tiny-gpt2.gguf --tokenize "Hello world" | grep -q "tokens\[2\] = 260 265"
	$(BIN) $(TEST_BUILD_DIR)/tiny-gpt2.gguf -p "Hi" --next-token-id --temperature 0 > $(TEST_BUILD_DIR)/actual.txt
	diff -u tests/expected.txt $(TEST_BUILD_DIR)/actual.txt
	@echo "All tests passed ($(BACKEND_DESC))."

clean:
	rm -rf $(BUILD_DIR)
