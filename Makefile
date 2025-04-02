CXX = gcc
CCC = nvcc

SRC_DIR = src
INCLUDE_DIR = include
TESTS_DIR = tests
BUILD_DIR = build

CXXFLAGS = -I./$(INCLUDE_DIR) -lm 
DEFINE_FLAGS = -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 -D CELL_CUTOFF_RADIUS_ANGST=10 -D UNIVERSE_LENGTH=100 -D TIME_RUN -D CELL_LENGTH_X=10 -D CELL_LENGTH_Y=10 -D CELL_LENGTH_Z=10

SIMULATIONS = naive nsquared nsquared_shared nsquared_n3l cell_list
TESTS = test_pdb_importer

.PHONY: tests run_tests clean md_simulations

md_simulations: $(SIMULATIONS)
tests: $(TESTS)

run_tests: tests
	$(BUILD_DIR)/test_pdb_importer $(TESTS_DIR)/input.pdb $(TESTS_DIR)/expected.pdb

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test_pdb_importer: $(TESTS_DIR)/test_pdb_importer.c $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/$@

naive: $(TESTS_DIR)/test_naive.c $(SRC_DIR)/naive.c $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEFINE_FLAGS) $^ -o $(BUILD_DIR)/$@

nsquared: $(SRC_DIR)/nsquared.cu $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CCC) $(CXXFLAGS) $(DEFINE_FLAGS) $^ -o $(BUILD_DIR)/$@

nsquared_shared: $(SRC_DIR)/nsquared_shared.cu $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CCC) $(CXXFLAGS) $(DEFINE_FLAGS) $^ -o $(BUILD_DIR)/$@

nsquared_n3l: $(SRC_DIR)/nsquared_n3l.cu $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CCC) $(CXXFLAGS) $(DEFINE_FLAGS) $^ -o $(BUILD_DIR)/$@

cell_list: $(SRC_DIR)/cell_list.cu $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CCC) $(CXXFLAGS) $(DEFINE_FLAGS) $^ -o $(BUILD_DIR)/$@

clean:
	rm $(BUILD_DIR)/*
