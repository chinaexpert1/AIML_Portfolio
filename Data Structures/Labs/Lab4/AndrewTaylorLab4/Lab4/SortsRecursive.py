'''Andrew Taylor
    atayl136
    This file is the recursive implementations of the required sorts, plus a random quicksort and
     a function to process the input and output reports.'''

# import statements for file handling and random numbers
import sys
import random
import os

# set a very high recursion limit, but memory ran out first
sys.setrecursionlimit(1000000)


# this is my insertion sort
def insertion_sort(arr, lowindex, highindex, comparisons=0, exchanges=0):
    for i in range(lowindex + 1, highindex + 1):
        key = arr[i]
        j = i - 1
        while j >= lowindex and arr[j] > key:
            arr[j + 1] = arr[j]
            j = j - 1
            comparisons += 1
            exchanges += 1
        arr[j + 1] = key
    return arr, comparisons, exchanges


# this is a quicksort that keeps track of comparisons and exchanges
def quicksort(arr, lowindex, highindex, comparisons=0, exchanges=0):
    # Stopping case
    if highindex <= lowindex:
        return arr, comparisons, exchanges

    if highindex - lowindex == 1:
        if arr[highindex] < arr[lowindex]:
            arr[lowindex], arr[highindex] = arr[highindex], arr[lowindex]
            exchanges += 1
        return arr, comparisons, exchanges

    pivot_index, comparisons, exchanges = partition(arr, lowindex, highindex, comparisons, exchanges)

    # Recursively sort elements before pivot and after pivot
    arr, comparisons, exchanges = quicksort(arr, lowindex, pivot_index - 1, comparisons, exchanges)
    arr, comparisons, exchanges = quicksort(arr, pivot_index + 1, highindex, comparisons, exchanges)
    return arr, comparisons, exchanges


# this creates a partition for the quicksort
def partition(arr, lowindex, highindex, comparisons=0, exchanges=0):
    pivot = arr[lowindex]
    i = lowindex + 1
    j = highindex

    while True:
        while i <= j and arr[i] <= pivot:
            i = i + 1
            comparisons += 1
        while arr[j] >= pivot and j >= i:
            j = j - 1
            comparisons += 1
        if j < i:
            break
        else:
            # Swap arr[i] and arr[j]
            arr[i], arr[j] = arr[j], arr[i]
            exchanges += 1

    # Swap arr[lowindex] with arr[j]
    arr[lowindex], arr[j] = arr[j], arr[lowindex]
    exchanges += 1
    return j, comparisons, exchanges



# this is a quicksort with the last 50 finished by an insertion sort
def quicksort50(arr, lowindex, highindex, comparisons=0, exchanges=0):
    if highindex <= lowindex:
        if highindex >= 0 and lowindex < len(arr) and arr[highindex] < arr[lowindex]:
            arr[lowindex], arr[highindex] = arr[highindex], arr[lowindex]
            exchanges += 1
        return arr, comparisons, exchanges

    if highindex - lowindex <= 50:
        return insertion_sort(arr, lowindex, highindex, comparisons, exchanges)

    pivot_index, comparisons, exchanges = partition(arr, lowindex, highindex, comparisons, exchanges)

    arr, comparisons, exchanges = quicksort(arr, lowindex, pivot_index - 1, comparisons, exchanges)
    arr, comparisons, exchanges = quicksort(arr, pivot_index + 1, highindex, comparisons, exchanges)
    return arr, comparisons, exchanges



# this is a quicksort with the last 100 finished by an insertion sort
def quicksort100(arr, lowindex, highindex, comparisons=0, exchanges=0):
    if highindex <= lowindex:
        if highindex >= 0 and lowindex < len(arr) and arr[highindex] < arr[lowindex]:
            arr[lowindex], arr[highindex] = arr[highindex], arr[lowindex]
            exchanges += 1
        return arr, comparisons, exchanges

    if highindex - lowindex <= 100:
        return insertion_sort(arr, lowindex, highindex, comparisons, exchanges)

    pivot_index, comparisons, exchanges = partition(arr, lowindex, highindex, comparisons, exchanges)

    arr, comparisons, exchanges = quicksort(arr, lowindex, pivot_index - 1, comparisons, exchanges)
    arr, comparisons, exchanges = quicksort(arr, pivot_index + 1, highindex, comparisons, exchanges)
    return arr, comparisons, exchanges



# this is a quicksort with the median-of-three for a pivot
def medianQuicksort(arr, lowindex, highindex, comparisons=0, exchanges=0):
    if highindex <= lowindex:
        if highindex >= 0 and lowindex < len(arr) and arr[highindex] < arr[lowindex]:
            arr[lowindex], arr[highindex] = arr[highindex], arr[lowindex]
            exchanges += 1
        return arr, comparisons, exchanges

    pivot_index, comparisons, exchanges = medianPartition(arr, lowindex, highindex, comparisons, exchanges)

    arr, comparisons, exchanges = medianQuicksort(arr, lowindex, pivot_index - 1, comparisons, exchanges)
    arr, comparisons, exchanges = medianQuicksort(arr, pivot_index + 1, highindex, comparisons, exchanges)
    return arr, comparisons, exchanges


# this creates a partition for the median-of-three quicksort
def medianPartition(arr, lowindex, highindex, comparisons=0, exchanges=0):
    pivot_index = median_of_three(arr, lowindex, highindex)
    pivot = arr[pivot_index]
    arr[pivot_index], arr[lowindex] = arr[lowindex], arr[pivot_index]

    i = lowindex + 1
    j = highindex

    while True:
        while i <= j and arr[i] <= pivot:
            i = i + 1
            comparisons += 1
        while arr[j] >= pivot and j >= i:
            j = j - 1
            comparisons += 1
        if j < i:
            break
        else:
            arr[i], arr[j] = arr[j], arr[i]
            exchanges += 1

    arr[lowindex], arr[j] = arr[j], arr[lowindex]
    exchanges += 1
    return j, comparisons, exchanges



# this calculates the median-of-three
def median_of_three(arr, lowindex, highindex):
    midindex = (lowindex + highindex) // 2
    if arr[lowindex] > arr[midindex]:
        arr[lowindex], arr[midindex] = arr[midindex], arr[lowindex]
    if arr[lowindex] > arr[highindex]:
        arr[lowindex], arr[highindex] = arr[highindex], arr[lowindex]
    if arr[midindex] > arr[highindex]:
        arr[midindex], arr[highindex] = arr[highindex], arr[midindex]
    return midindex


# this is a node class for the linked implementation below
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


# this is a natural merge sort
def natural_merge_sort(arr, comparisons=0, exchanges=0):
    comparisons = 0
    exchanges = 0

    def merge(node1, node2):
        nonlocal comparisons, exchanges
        merged_head = merged_tail = None

        while node1 and node2:
            comparisons += 1
            if node1.value <= node2.value:
                if not merged_tail:
                    merged_head = merged_tail = node1
                else:
                    merged_tail.next = node1
                    merged_tail = node1
                node1 = node1.next
            else:
                if not merged_tail:
                    merged_head = merged_tail = node2
                else:
                    merged_tail.next = node2
                    merged_tail = node2
                node2 = node2.next
                exchanges += 1

        merged_tail.next = node1 or node2
        return merged_head

    def find_part(current_node):
        if not current_node.next:
            return current_node, None

        part_end = current_node
        while part_end.next and part_end.value <= part_end.next.value:
            part_end = part_end.next

        next_part_start = part_end.next
        part_end.next = None
        return current_node, next_part_start

    first_node = Node(arr[0])
    current_node = first_node
    for value in arr[1:]:
        current_node.next = Node(value)
        current_node = current_node.next

    part_list = []
    current_start = first_node
    while current_start:
        part, current_start = find_part(current_start)
        part_list.append(part)

    while len(part_list) > 1:
        merged_parts = []
        i = 0
        while i < len(part_list) - 1:
            merged_part = merge(part_list[i], part_list[i + 1])
            merged_parts.append(merged_part)
            i += 2
        if i < len(part_list):
            merged_parts.append(part_list[i])
        part_list = merged_parts

    sorted_list = []
    current_node = part_list[0]
    while current_node:
        sorted_list.append(current_node.value)
        current_node = current_node.next

    return sorted_list, comparisons, exchanges



# I added a quicksort using a random pivot to compare
def random_pivot_quicksort(arr, low, high):
    comparisons = 0
    exchanges = 0
    # creates partition for a random pivot implementation
    def random_partition(arr, low, high):
        nonlocal exchanges
        pivot_index = random.randint(low, high)
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        exchanges += 1
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            nonlocal comparisons
            comparisons += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                exchanges += 1
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        exchanges += 1
        return i + 1
    # inner function
    def quicksort(arr, low, high):
        if low < high:
            pivot_index = random_partition(arr, low, high)
            quicksort(arr, low, pivot_index - 1)
            quicksort(arr, pivot_index + 1, high)

    quicksort(arr, low, high)
    return arr, comparisons, exchanges


# this reads in a .dat file and parses the numbers
def read_numbers(filename):
    with open(filename, 'r') as file:
        numbers = [int(num) for line in file.readlines() for num in line.strip().split()]
    return numbers


# this is the process function to read the input, sort the arrays, and output the reports
def process_files_all_sorts_print(output_filename):
    sorting_algorithms = [natural_merge_sort, quicksort100, quicksort50, random_pivot_quicksort, medianQuicksort, quicksort]
    sorting_results = []

    sorted_arr = []  # Initialize the variable outside the loops

    for filename in os.listdir('input_files'):
        if filename.endswith(".dat"):
            input_file = os.path.join('input_files', filename)

            for sorting_algorithm in sorting_algorithms:
                arr = read_numbers(input_file)
                sorted_arr, comparisons, exchanges = sorting_algorithm(arr, 0, len(arr) - 1)
                sorting_results.append((sorting_algorithm.__name__, filename, comparisons, exchanges))


    with open(output_filename, 'w') as file:
        for result in sorting_results:
            output_string = f"Algorithm: {result[0]}\nFile: {result[1]}\nComparisons: {result[2]}\nExchanges: {result[3]}\n\n"
            file.write(output_string)
            print(output_string)

    sorted_arr = []  # Initialize the variable outside the loops
    sorting_results = []
    sorting_algorithms = [natural_merge_sort, quicksort100, quicksort50, random_pivot_quicksort, medianQuicksort, quicksort]

    for filename in os.listdir('size_50_files'):
        if filename.endswith(".dat"):
            input_file = os.path.join('size_50_files', filename)



            for sorting_algorithm in sorting_algorithms:
                arr = read_numbers(input_file)
                sorted_arr, comparisons, exchanges = sorting_algorithm(arr, 0, len(arr) - 1)
                sorting_results.append((sorting_algorithm.__name__, filename, comparisons, exchanges))

            print(sorting_results)

            for result in sorting_results:
                # Write results to a separate output file for each algorithm
                output_filename = f"{result[0]}_{filename[:-4]}_output.txt"
                with open(output_filename, 'w') as file:
                    output_string = f"Algorithm: {result[0]}\nFile: {result[1]}\nComparisons: {result[2]}\nExchanges: {result[3]}\n\n"
                    file.write(output_string)
                    print(output_string)
                    file.write(f'sorted array: {sorted_arr}\n')


# I chose a filename for the combined report
output_filename = "all_sorts_output_print.txt"

# this runs the program, very simple
process_files_all_sorts_print(output_filename)












