# Makefile for Tensor class
CXX = nvcc
CXXFLAGS = -std=c++11 -O2
TARGET = test_tensor
SOURCES = Tensor.cu test_tensor.cu

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# 运行测试
test: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET)

# 帮助信息
help:
	@echo "可用的命令:"
	@echo "  make          - 编译程序"
	@echo "  make test     - 编译并运行测试"
	@echo "  make clean    - 清理编译文件"
	@echo "  make help     - 显示此帮助信息"

.PHONY: all test clean help