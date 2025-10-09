# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++11 -O2 -arch=sm_50
CXX_FLAGS = -std=c++11 -O2

# Source files
TENSOR_SRC = Tensor.cu
ACTIVATION_SRC = ActivationFunctions.cu
TEST_ACTIVATION_SRC = test_activations.cu
TEST_TENSOR_SRC = test_tensor.cu

# Object files
TENSOR_OBJ = Tensor.o
ACTIVATION_OBJ = ActivationFunctions.o
TEST_ACTIVATION_OBJ = test_activations.o
TEST_TENSOR_OBJ = test_tensor.o

# Executables
TEST_ACTIVATION_EXEC = test_activations
TEST_TENSOR_EXEC = test_tensor

# Default target
all: $(TEST_ACTIVATION_EXEC) $(TEST_TENSOR_EXEC)

# Build activation test executable
$(TEST_ACTIVATION_EXEC): $(TENSOR_OBJ) $(ACTIVATION_OBJ) $(TEST_ACTIVATION_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Build tensor test executable
$(TEST_TENSOR_EXEC): $(TENSOR_OBJ) $(TEST_TENSOR_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile Tensor object
$(TENSOR_OBJ): $(TENSOR_SRC) Tensor.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile ActivationFunctions object
$(ACTIVATION_OBJ): $(ACTIVATION_SRC) ActivationFunctions.h Tensor.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile activation test object
$(TEST_ACTIVATION_OBJ): $(TEST_ACTIVATION_SRC) ActivationFunctions.h Tensor.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile tensor test object
$(TEST_TENSOR_OBJ): $(TEST_TENSOR_SRC) Tensor.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f *.o $(TEST_ACTIVATION_EXEC) $(TEST_TENSOR_EXEC)

# Run all tests
test: $(TEST_ACTIVATION_EXEC) $(TEST_TENSOR_EXEC)
	@echo "Running Tensor tests..."
	./$(TEST_TENSOR_EXEC)
	@echo "\nRunning Activation Function tests..."
	./$(TEST_ACTIVATION_EXEC)

# Run tensor tests only
test-tensor: $(TEST_TENSOR_EXEC)
	./$(TEST_TENSOR_EXEC)

# Run activation tests only
test-activations: $(TEST_ACTIVATION_EXEC)
	./$(TEST_ACTIVATION_EXEC)

# Run with memory check (if valgrind available)
memcheck-tensor: $(TEST_TENSOR_EXEC)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TEST_TENSOR_EXEC)

memcheck-activations: $(TEST_ACTIVATION_EXEC)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TEST_ACTIVATION_EXEC)

# Phony targets
.PHONY: all clean test test-tensor test-activations memcheck-tensor memcheck-activations