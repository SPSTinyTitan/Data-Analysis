BIN := particle

SRC_DIR := src
OBJ_DIR := build

IDIR := include
LDIR := lib

CC := nvcc
NVCC := nvcc
CFLAGS :=  -I$(IDIR) -I$(SRC_DIR) -arch=sm_35 -rdc=true --extended-lambda
LDFLAGS := -L$(LDIR) -arch=sm_35
LDLIBS = -lcusolver -lcublas 

SRC := $(wildcard $(SRC_DIR)/*.cu)
OBJ := $(SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir $@

clean:
	$(RM) -r $(OBJ_DIR)

.PHONY: all clean
