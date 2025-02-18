CC = gcc
CFLAGS = -I./include

TESTS_DIR = tests
SRC_DIR = src
BUILD_DIR = build

.PHONY: tests run_tests clean

TESTS = test_pdb_importer test_naive #test_cell_list

tests: $(TESTS)

run_tests: tests
	$(BUILD_DIR)/test_pdb_importer $(TESTS_DIR)/input.pdb $(TESTS_DIR)/expected.pdb
	$(BUILD_DIR)/test_naive $(TESTS_DIR)/input.pdb

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test_pdb_importer: $(TESTS_DIR)/test_pdb_importer.c $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

test_naive: $(TESTS_DIR)/test_naive.c $(SRC_DIR)/naive.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

#test_cell_list: $(SRC_DIR)/md.cu $(SRC_DIR)/pdb_importer.c | $(BUILD_DIR)
#	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm $(BUILD_DIR)/*
