CC = gcc
CFLAGS = -I./include

TESTS_DIR = tests
SRC_DIR = src

.PHONY: tests run_tests clean

TESTS = test_pdb_importer

tests: $(TESTS)

run_tests: tests
	./test_pdb_importer $(TESTS_DIR)/input.pdb $(TESTS_DIR)/expected.pdb

test_pdb_importer: $(TESTS_DIR)/test_pdb_importer.c $(SRC_DIR)/pdb_importer.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm test_pdb_importer