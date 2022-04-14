import random
import time
import tracemalloc
import os

DECIMAL_PRECISION = 4
RANDOMINT_LOWERBOUND = 0
RANDOMINT_UPPERBOUND = 9
RUNS_PER_DIMENSION = 3
DIMENSION_START = 50
DIMENSION_INCREMENT = 10
DIMENSION_END = 100
AVG_INDEX = RUNS_PER_DIMENSION
STDEV_INDEX = AVG_INDEX + 1
MEMORY_PROFILING_ENABLED = False
OUTPUT_DIRECTORY = "output"


def standard_matrix_multiply(matrix1, matrix2, dimension):
    result_matrix = []
    for i in range(0, dimension):
        result_matrix.append([])
        for j in range(0, dimension):
            result_matrix[i].append(0)

    print("Matrix 1:")
    print_matrix(matrix1)
    print("Matrix 2:")
    print_matrix(matrix2)

    print("\nCalculating product using standard algorithm...")
    # standard matrix multiplication algorithm
    start_time = time.perf_counter()
    for i in range(len(matrix1)):  # go through rows of 1
        for j in range(len(matrix2[0])):  # go through columns of 2
            for k in range(len(matrix2)):  # go through rows of 2
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
    calc_time = time.perf_counter() - start_time
    print("Solution matrix:")
    print_matrix(result_matrix)
    print(f' Calculated in {calc_time} seconds for dimensions {dimension}x{dimension}')
    return [result_matrix, calc_time]


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


def generate_matrix(dimension):
    matrix1 = []
    for i in range(0, dimension):
        matrix1.append([])
        for j in range(0, dimension):
            matrix1[i].append(random.randint(RANDOMINT_LOWERBOUND, RANDOMINT_UPPERBOUND))
    return matrix1


def build_data_container():
    container = {}
    d = DIMENSION_START
    while d <= DIMENSION_END:
        container[d] = []  # key = dimension of matrices for that run
        for i in range(0, RUNS_PER_DIMENSION+2):  # extra 2 spaces for storing average and standard deviation
            container[d].append(0)
        d += DIMENSION_INCREMENT
    return container


def compute_run_statistics(data):
    for key in data.keys():
        average = 0
        stdev = 0
        for n in data[key]:
            average += n
        average = average / RUNS_PER_DIMENSION

        for i in range(0, RUNS_PER_DIMENSION):
            stdev += (data[key][i] - average)**2
        stdev = (stdev / RUNS_PER_DIMENSION)**(1/2)
        data[key][AVG_INDEX] = average
        data[key][STDEV_INDEX] = stdev


def strassen_matrix_multiply(matrix1, matrix2, dimension):
    pass


def print_data_report(data):
    for key in data.keys():
        print(f'Dimension {key}: {data[key][0]} seconds')


def print_data_report_CSV(data, filename):
    filepath = os.path.join(OUTPUT_DIRECTORY, filename)
    title_row = ["Dimension"]
    for i in range(0, RUNS_PER_DIMENSION):
        title_row.append(f"Run{i+1}")
    title_row.extend(["Average", "Standard Deviation"])

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(','.join(title_row))
        for key in data.keys():
            f.write('\n')
            row_array = [str(key)]
            for i in data[key]:
                row_array.append(str(i))
            f.write(','.join(row_array))
        print(f"Data written to {filepath}.")


if __name__ == "__main__":
    dimension = DIMENSION_START
    data_standard_space = build_data_container()
    data_standard_time = build_data_container()
    data_strassen_space = build_data_container()
    data_strassen_time = build_data_container()

    while dimension <= DIMENSION_END:
        for run in range(0, RUNS_PER_DIMENSION):
            matrix1 = generate_matrix(dimension)  # unique matrices for each run
            matrix2 = generate_matrix(dimension)
            if MEMORY_PROFILING_ENABLED:
                tracemalloc.stop()
                tracemalloc.start()
                standard_matrix_multiply(matrix1, matrix2, dimension)
                stats = tracemalloc.get_traced_memory()
                memory_usage_stats_KB = (stats[0]/1000, stats[1]/1000)  # converting bytes to Kilobytes
                print("Used %dKB of memory" % memory_usage_stats_KB[1])
            else:
                t = round(standard_matrix_multiply(matrix1, matrix2, dimension)[1], DECIMAL_PRECISION)
                data_standard_time[dimension][run] = t

        dimension += DIMENSION_INCREMENT
    compute_run_statistics(data_standard_time)
    print(data_standard_time)
    print_data_report_CSV(data_standard_time, "test.csv")
