import random
import time
import tracemalloc
import os
import numpy

DECIMAL_PRECISION = 4  # Controls the floating point precision for time and memory usage statistics
RANDOMINT_LOWERBOUND = 0  # Controls the range of numbers randomized numbers selected for elements of the matrices
RANDOMINT_UPPERBOUND = 9
RUNS_PER_DIMENSION = 1
DIMENSION_START = 50
DIMENSION_INCREMENT = 50
DIMENSION_END = 750
AVG_INDEX = RUNS_PER_DIMENSION
STDEV_INDEX = AVG_INDEX + 1
MEMORY_PROFILING_ENABLED = True  # Enable or disable memory usage analysis, since it incurs significant overhead
LEAF_SIZE = 800  # equivalent to the k-value/Small Problem Cutoff
OUTPUT_DIRECTORY = "output"  # Name of the directory to be created for output of run data
BLAS_OVERRIDE = False  # Used to force Numpy's built in matrix multiplication
BINARY_DIMENSIONS_ENABLED = False  # Enable to only test matrices with dimensions equal to a power of 2 between the start and end bounds
FRACTIONAL_LEAF_SIZE = 0.10  # Fractional K-value
INDEPENDENT_SMC_OVERRIDE = True  #Force testing for only Strassen's algorithm with LEAF_SIZE small problem cutoff


def standard_matrix_multiply(matrix1, matrix2, dimension):
    result_matrix = []

    # Generate matrix of given dimensions prefilled with zeros
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
    if BLAS_OVERRIDE:  # Use numpy Matrix multiplication instead
        result_matrix = numpy.matmul(matrix1, matrix2)
    else:
        for i in range(len(matrix1)):  # go through rows of 1
            for j in range(len(matrix2[0])):  # go through columns of 2
                for k in range(len(matrix2)):  # go through rows of 2
                    result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
    calc_time = time.perf_counter() - start_time  # Get elapsed time
    print("Solution matrix:")
    print_matrix(result_matrix)
    print(f' Calculated in {calc_time} seconds for dimensions {dimension}x{dimension}')
    return [result_matrix, calc_time]


def standard_matrix_multiply_kurz(matrix1, matrix2, dimension):
    """
    Identical to above function, but with logging and performance analysis
    removed, optimal for calling from a recursive context (Strassens with small problem cutoff)
    :param matrix1:
    :param matrix2:
    :param dimension:
    :return:
    """

    result_matrix = []

    # Generate matrix of given dimensions prefilled with zeros
    for i in range(0, dimension):
        result_matrix.append([])
        for j in range(0, dimension):
            result_matrix[i].append(0)
    # standard matrix multiplication algorithm
    if BLAS_OVERRIDE:
        result_matrix = numpy.matmul(matrix1, matrix2)
    else:
        for i in range(len(matrix1)):  # go through rows of 1
            for j in range(len(matrix2[0])):  # go through columns of 2
                for k in range(len(matrix2)):  # go through rows of 2
                    result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return [result_matrix, None]


def print_matrix(matrix):
    rlen = len(matrix)
    clen = len(matrix[0])

    if rlen < 8 and clen < 8:  # avoid printing very large matrices
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                print(matrix[i][j], ' ', end='')
            print()
        print()
    else:
        print(" [Too large to display]")


def generate_matrix(dimension):
    matrix1 = []

    # create matrix represented as 2d array
    for i in range(0, dimension):
        matrix1.append([])
        for j in range(0, dimension):
            #Fill each element with a random number
            matrix1[i].append(random.randint(RANDOMINT_LOWERBOUND, RANDOMINT_UPPERBOUND))
    return matrix1


def build_data_container():
    """
    Builds a data container to store per-run statistics for time and memory usage.
    :return:
    """
    container = {}  # Container starts as an empty dictionary
    d = DIMENSION_START
    while d <= DIMENSION_END:
        container[d] = []  # dictionary key = dimension of matrices for that run, each key holds an array to store data
        for i in range(0, RUNS_PER_DIMENSION+2):  # reserve last 2 spaces for storing average and standard deviation
            container[d].append(0)
        if BINARY_DIMENSIONS_ENABLED:
            d *= 2
        else:
            d += DIMENSION_INCREMENT
    return container


def compute_run_statistics(data):
    for key in data.keys():
        average = 0
        stdev = 0
        for n in range(0, RUNS_PER_DIMENSION):
            average += data[key][n]
        average = average / RUNS_PER_DIMENSION

        # Calculation of standard deviation
        for i in range(0, RUNS_PER_DIMENSION):
            stdev += (data[key][i] - average)**2  # Calculating variance
        stdev = (stdev / RUNS_PER_DIMENSION)**(1/2)  # Get the square root of the variance to obtain the standard deviation
        data[key][AVG_INDEX] = round(average, DECIMAL_PRECISION)
        data[key][STDEV_INDEX] = stdev


def strassen_matrix_multiply(matrix1, matrix2, dimension, cutoff = LEAF_SIZE):
    """
    Wraps call to recursive algorithm to simplify performance analysis and logging
    :param matrix1:
    :param matrix2:
    :return:
    """
    print(f"\nCalculating product using strassen algorithm (LEAF_SIZE = {cutoff})...")
    start_time = time.perf_counter()
    result_matrix = strassen_matrix_multiply_recursive(matrix1, matrix2, cutoff=cutoff)
    calc_time = time.perf_counter() - start_time  # Store total runtime
    print("Strassen matrix:")
    print_matrix(result_matrix)
    print(f' Calculated in {calc_time} seconds for dimensions {dimension}x{dimension}')
    return [result_matrix, calc_time]

def strassen_matrix_multiply_recursive(matrix1, matrix2, cutoff = LEAF_SIZE):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """

    # Base case when size of matrices is 1x1
    if len(matrix1) == 1:
        return matrix1 * matrix2

    # small problem cutoff
    elif len(matrix1) <= cutoff or BLAS_OVERRIDE:
        # return numpy.matmul(matrix1, matrix2)

        # print(standard_matrix_multiply(matrix1, matrix2, len(matrix1))[0])
        return numpy.array(standard_matrix_multiply_kurz(matrix1, matrix2, len(matrix1))[0])

    # need to pad matrices with uneven dimensions for strassen to work correctly
    # https://cs.stackexchange.com/questions/97998/strassens-matrix-multiplication-algorithm-when-n-is-not-a-power-of-2
    elif len(matrix1) % 2 != 0:
        matrix1 = numpy.pad(matrix1, [(0, 1), (0, 1)])
        matrix2 = numpy.pad(matrix2, [(0, 1), (0, 1)])


    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(matrix1)
    e, f, g, h = split(matrix2)

    # Computing the 7 products, recursively (p1, p2...p7)
    # print("a")
    # print(a)
    # print("f")
    # print(f)
    # print("h")
    # print(h)
    # print(numpy.subtract(f,h))
    p1 = strassen_matrix_multiply_recursive(a, f - h, cutoff=cutoff)
    p2 = strassen_matrix_multiply_recursive(a + b, h, cutoff=cutoff)
    p3 = strassen_matrix_multiply_recursive(c + d, e, cutoff=cutoff)
    p4 = strassen_matrix_multiply_recursive(d, g - e, cutoff=cutoff)
    p5 = strassen_matrix_multiply_recursive(a + d, e + h, cutoff=cutoff)
    p6 = strassen_matrix_multiply_recursive(b - d, g + h, cutoff=cutoff)
    p7 = strassen_matrix_multiply_recursive(a - c, e + f, cutoff=cutoff)

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    # This is done by vertically stacking 2 horizontally stacked matrices
    c = numpy.vstack((numpy.hstack((c11, c12)), numpy.hstack((c21, c22))))
    return c


def split(matrix):
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape  # Get the shape of the numpy matrix and store the row and column values
    row2, col2 = row//2, col//2  # Get the midpoint values for the row and column dimensions
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]


def print_data_report(data):
    for key in data.keys():
        print(f'Dimension {key}: {data[key][0]} seconds')


def print_data_report_CSV(data, filename):
    filepath = os.path.join(OUTPUT_DIRECTORY, filename) #Generate proper filepath regardless of platform
    title_row = ["Dimension"]
    for i in range(0, RUNS_PER_DIMENSION):
        title_row.append(f"Run{i+1}")  # Add a row for each run
    title_row.extend(["Average", "Standard Deviation"])

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)  # Create output directory only if it doesnt already exist
    with open(filepath, 'w') as f:
        f.write(','.join(title_row))  # write title row to top of file
        for key in data.keys():  # Iterate through every dimension in the data container
            f.write('\n')
            row_array = [str(key)]  # Create 1 element array starting with the dimension from the data container
            for i in data[key]:
                row_array.append(str(i))  # add every element from the array corresponding to each dimension in the data container
            f.write(','.join(row_array))  # write the array to the file in one line, separated by commas
        print(f"Data written to {filepath}.")


if __name__ == "__main__":
    dimension = DIMENSION_START  # Initialize dimension value for main loop
    data_standard_space = build_data_container()
    data_standard_time = build_data_container()
    data_strassen_space = build_data_container()
    data_strassen_time = build_data_container()
    data_strassen_time_frac = build_data_container()
    data_strassen_space_frac = build_data_container()

    while dimension <= DIMENSION_END:
        for run in range(0, RUNS_PER_DIMENSION):
            matrix1 = numpy.array(generate_matrix(dimension))  # unique matrices for each run, converted to numpy arrays
            matrix2 = numpy.array(generate_matrix(dimension))

            if INDEPENDENT_SMC_OVERRIDE:
                # Testing for Strassen algorithm with fixed small problem cutoff
                #tracemalloc.start()
                t_strassen = round(standard_matrix_multiply(matrix1, matrix2, dimension)[1], DECIMAL_PRECISION)
                #mem_usage = tracemalloc.get_traced_memory()
                #tracemalloc.stop()
                data_strassen_time[dimension][run] = t_strassen
                #data_strassen_space[dimension][run] = mem_usage[1] / 1000

            elif MEMORY_PROFILING_ENABLED :
                # Testing for standard algorithm
                tracemalloc.start()
                t_standard = round(standard_matrix_multiply(matrix1, matrix2, dimension)[1], DECIMAL_PRECISION)
                mem_usage = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                data_standard_time[dimension][run] = t_standard
                data_standard_space[dimension][run] = mem_usage[1]/1000  # Memory usage given in bytes, so divide by 1000 to conver to kilobytes

                # Testing for Strassen algorithm with fixed small problem cutoff
                tracemalloc.start()
                t_strassen = round(strassen_matrix_multiply(matrix1, matrix2, dimension)[1], DECIMAL_PRECISION)
                mem_usage = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                data_strassen_time[dimension][run] = t_strassen
                data_strassen_space[dimension][run] = mem_usage[1]/1000

                # Testing for Strassen algorithm with fractional small problem cutoff
                tracemalloc.start()
                cutoff = int(dimension*FRACTIONAL_LEAF_SIZE)
                t_strassen_frac = round(strassen_matrix_multiply(matrix1, matrix2, dimension, cutoff=cutoff)[1], DECIMAL_PRECISION)
                mem_usage = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                data_strassen_time_frac[dimension][run] = t_strassen_frac
                data_strassen_space_frac[dimension][run] = mem_usage[1] / 1000

                # Compute average and standard deviation for all data containers in-place
                compute_run_statistics(data_standard_time)
                compute_run_statistics(data_strassen_time)
                compute_run_statistics(data_strassen_time_frac)
                compute_run_statistics(data_standard_space)
                compute_run_statistics(data_strassen_space)
                compute_run_statistics(data_strassen_space_frac)
                print(data_standard_time)

                #Print all data containers to csv files
                print_data_report_CSV(data_standard_time, "standard_algo_time.csv")
                print_data_report_CSV(data_strassen_time, "strassen_algo_time.csv")
                print_data_report_CSV(data_strassen_time_frac, "strassen_algo_time_frac.csv")
                print_data_report_CSV(data_standard_space, "standard_algo_space.csv")
                print_data_report_CSV(data_strassen_space, "strassen_algo_space.csv")
                print_data_report_CSV(data_strassen_space_frac, "strassen_algo_space_frac.csv")

                # standard_matrix_multiply(matrix1, matrix2, dimension)
                # stats = tracemalloc.get_traced_memory()
                # memory_usage_stats_KB = (stats[0]/1000, stats[1]/1000)  # converting bytes to Kilobytes
                # print("Used %dKB of memory" % memory_usage_stats_KB[1])
            else:
                # Testing for standard algorithm
                t_standard = round(standard_matrix_multiply(matrix1, matrix2, dimension)[1], DECIMAL_PRECISION)
                data_standard_time[dimension][run] = t_standard

                # Testing for Strassen algorithm with fixed small problem cutoff
                t_strassen = round(strassen_matrix_multiply(matrix1, matrix2, dimension)[1], DECIMAL_PRECISION)
                data_strassen_time[dimension][run] = t_strassen


                # Testing for Strassen algorithm with fractional small problem cutoff
                cutoff = int(dimension * FRACTIONAL_LEAF_SIZE)
                t_strassen_frac = round(strassen_matrix_multiply(matrix1, matrix2, dimension, cutoff=cutoff)[1],
                                        DECIMAL_PRECISION)
                data_strassen_time_frac[dimension][run] = t_strassen_frac

                # Compute average and standard deviation for all data containers in-place
                compute_run_statistics(data_standard_time)
                compute_run_statistics(data_strassen_time)
                compute_run_statistics(data_strassen_time_frac)
                compute_run_statistics(data_standard_space)
                compute_run_statistics(data_strassen_space)
                compute_run_statistics(data_strassen_space_frac)
                print(data_standard_time)

                # Print all data containers to csv files
                print_data_report_CSV(data_standard_time, "standard_algo_time.csv")
                print_data_report_CSV(data_strassen_time, "strassen_algo_time.csv")
                print_data_report_CSV(data_strassen_time_frac, "strassen_algo_time_frac.csv")
                print_data_report_CSV(data_standard_space, "standard_algo_space.csv")
                print_data_report_CSV(data_strassen_space, "strassen_algo_space.csv")
                print_data_report_CSV(data_strassen_space_frac, "strassen_algo_space_frac.csv")

        if BINARY_DIMENSIONS_ENABLED:
            dimension *= 2
        else:
            dimension += DIMENSION_INCREMENT

