CXX = cc

SRC_DIR = src
INCLUDE_DIR = include
TESTS_DIR = tests
BUILD_DIR = build

CCC = nvcc

CXXFLAGS = -I./$(INCLUDE_DIR)

SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS)) $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

.PHONY: tests run_tests clean md_simulations

TESTS = test_pdb_importer #test_naive #test_cell_list

SIMULATIONS = nsquared cell_list

tests: $(TESTS)

md_simulations: $(SIMULATIONS)

run_tests: tests
	$(BUILD_DIR)/test_pdb_importer $(TESTS_DIR)/input.pdb $(TESTS_DIR)/expected.pdb
#	$(BUILD_DIR)/test_naive $(TESTS_DIR)/input.pdb
#
#



$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test_pdb_importer: $(TESTS_DIR)/test_pdb_importer.c $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/$@

#test_naive: $(TESTS_DIR)/test_naive.c $(SRC_DIR)/naive.c | $(BUILD_DIR)
#	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

#test_cell_list: $(SRC_DIR)/md.cu $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
#	$(CC) $(CFLAGS) $^ -o $@
#
nsquared: $(SRC_DIR)/pdb_importer.c $(SRC_DIR)/nsquared.cu | $(BUILD_DIR)
	$(CCC) $(CXXFLAGS) -D TIMESTEPS=10 -D TIMESTEP_DURATION=1 $^ -o $(BUILD_DIR)/$@

cell_list: $(SRC_DIR)/pdb_importer.c $(SRC_DIR)/cell_list.cu | $(BUILD_DIR)
	$(CCC) $(CXXFLAGS) -g $^ -o $(BUILD_DIR)/$@

clean:
	rm $(BUILD_DIR)/*
