import random
import time
import tracemalloc

RANDOMINT_LOWERBOUND = 0
RANDOMINT_UPPERBOUND = 9
MEMORY_PROFILING_ENABLED = True


def standard_matrix_multiply(dimension):
    matrix1 = []
    matrix2 = []
    matrix3 = []

    for i in range(0, dimension):
        matrix1.append([])
        matrix2.append([])
        matrix3.append([])
        for j in range(0, dimension):
            matrix1[i].append(random.randint(RANDOMINT_LOWERBOUND, RANDOMINT_UPPERBOUND))
            matrix2[i].append(random.randint(RANDOMINT_LOWERBOUND, RANDOMINT_UPPERBOUND))
            matrix3[i].append(0)

    print("Matrix 1:")
    print_matrix(matrix1)
    print("Matrix 2:")
    print_matrix(matrix2)

    print("\nCalculating product using standard algorithm...")
    # standard matrix multiplication algorithm
    start_time = time.perf_counter()
    for i in range(len(matrix1)): # go through rows of 1
        for j in range(len(matrix2[0])): # go through columns of 2
            for k in range(len(matrix2)): # go through rows of 2\
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j]
    calc_time = time.perf_counter() - start_time
    print("Solution matrix:")
    print_matrix(matrix3)
    print(f'Calculated in {calc_time} seconds')


def print_matrix(matrix):
    rlen = len(matrix)
    clen = len(matrix[0])

    if rlen < 6 and clen < 6:  # avoid printing very large matrices
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                print(matrix[i][j], ' ', end='')
            print()
        print()
    else:
        print(" [Too large to display]")


if __name__ == "__main__":
    dimension = 100
    if MEMORY_PROFILING_ENABLED:
        tracemalloc.start()
        standard_matrix_multiply(dimension)
        stats = tracemalloc.get_traced_memory()
        memory_usage_stats_KB = (stats[0]/1000, stats[1]/1000) #converting bytes to Kilobytes
        print("Used %dKB of memory" % memory_usage_stats_KB[1])
    else:
        standard_matrix_multiply(dimension)
