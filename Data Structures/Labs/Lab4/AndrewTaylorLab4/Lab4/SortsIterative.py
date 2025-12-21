'''Andrew Taylor
    atayl136
    This is the iterative solution to the sorting problems, plus a random quicksort and
     a process function at the end to print and write statements to file.'''

# import statements for file handling and random numbers
import sys
import random
import os


# this is a standard insertion sort
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

# this function creates a partition for quicksort
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

# this is the iterative version of quicksort0
def quicksort_iterative(arr, lowindex, highindex, comparisons=0, exchanges=0):
    stack = [(lowindex, highindex)]

    while stack:
        low, high = stack.pop()

        if high <= low:
            continue

        if high - low == 1:
            if arr[high] < arr[low]:
                arr[low], arr[high] = arr[high], arr[low]
                exchanges += 1
            continue

        pivot_index, comparisons, exchanges = partition(arr, low, high, comparisons, exchanges)

        stack.append((low, pivot_index - 1))
        stack.append((pivot_index + 1, high))

    return arr, comparisons, exchanges


# this is quicksort finished off the last 50 with insertion sort
def quicksort50_iterative(arr, lowindex, highindex, comparisons=0, exchanges=0):
    stack = [(lowindex, highindex)]

    while stack:
        low, high = stack.pop()

        if high <= low:
            if high >= 0 and low < len(arr) and arr[high] < arr[low]:
                arr[low], arr[high] = arr[high], arr[low]
                exchanges += 1
            continue

        if high - low <= 50:
            arr, comparisons, exchanges = insertion_sort(arr, low, high, comparisons, exchanges)
            continue

        pivot_index, comparisons, exchanges = partition(arr, low, high, comparisons, exchanges)

        stack.append((low, pivot_index - 1))
        stack.append((pivot_index + 1, high))

    return arr, comparisons, exchanges


# this is quicksort finished off the last 100 with insertion sort
def quicksort100_iterative(arr, lowindex, highindex, comparisons=0, exchanges=0):
    stack = [(lowindex, highindex)]

    while stack:
        low, high = stack.pop()

        if high <= low:
            if high >= 0 and low < len(arr) and arr[high] < arr[low]:
                arr[low], arr[high] = arr[high], arr[low]
                exchanges += 1
            continue

        if high - low <= 100:
            arr, comparisons, exchanges = insertion_sort(arr, low, high, comparisons, exchanges)
            continue

        pivot_index, comparisons, exchanges = partition(arr, low, high, comparisons, exchanges)

        stack.append((low, pivot_index - 1))
        stack.append((pivot_index + 1, high))

    return arr, comparisons, exchanges



# this creates a partition for a median-of-three quicksort
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


# this is an iterative version of a median-of-three quicksort
def medianQuicksort_iterative(arr, lowindex, highindex, comparisons=0, exchanges=0):
    stack = [(lowindex, highindex)]

    while stack:
        low, high = stack.pop()

        if high <= low:
            if high >= 0 and low < len(arr) and arr[high] < arr[low]:
                arr[low], arr[high] = arr[high], arr[low]
                exchanges += 1
            continue

        pivot_index, comparisons, exchanges = medianPartition(arr, low, high, comparisons, exchanges)

        stack.append((low, pivot_index - 1))
        stack.append((pivot_index + 1, high))

    return arr, comparisons, exchanges


# nodes for linked implementation
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node



# this is an iterative version of a Natural Merge sort
def natural_merge_sort_iterative(arr, comparisons=0, exchanges=0):
    def find_part(current_node):
        if not current_node.next:
            return current_node, None

        part_end = current_node
        while part_end.next and part_end.value <= part_end.next.value:
            part_end = part_end.next

        next_part_start = part_end.next
        part_end.next = None
        return current_node, next_part_start

    # merge inner function
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

    first_node = Node(arr[0])
    current_node = first_node
    for value in arr[1:]:
        current_node.next = Node(value)
        current_node = current_node.next

    queue = []
    current_start = first_node
    while current_start:
        part, current_start = find_part(current_start)
        queue.append(part)

    while len(queue) > 1:
        node1 = queue.pop(0)
        node2 = queue.pop(0)
        merged_part = merge(node1, node2)
        queue.insert(0, merged_part)

    sorted_list = []
    current_node = queue[0]
    while current_node:
        sorted_list.append(current_node.value)
        current_node = current_node.next

    return sorted_list, comparisons, exchanges


# this is an iterative version of a quicksort using a random pivot
def random_pivot_quicksort_iterative(arr, low, high, comparisons=0, exchanges=0):
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

    stack = [(low, high)]

    while stack:
        low, high = stack.pop()

        if low < high:
            pivot_index = random_partition(arr, low, high)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))

    return arr, comparisons, exchanges



# this function reads in a .dat file and parses it
def read_numbers(filename):
    with open(filename, 'r') as file:
        numbers = [int(num) for line in file.readlines() for num in line.strip().split()]
    return numbers


# this is the process function that runs the sorts aned prints/writes the reports
def process_files_all_sorts_print(output_filename):
    sorting_algorithms = [natural_merge_sort_iterative, quicksort100_iterative, quicksort50_iterative, random_pivot_quicksort_iterative, medianQuicksort_iterative, quicksort_iterative]
    sorting_results = []

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

    sorted_arr = []
    sorting_results = []
    sorting_algorithms = [natural_merge_sort_iterative, quicksort100_iterative, quicksort50_iterative, random_pivot_quicksort_iterative, medianQuicksort_iterative, quicksort_iterative]

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

# set a file name for the combined report
output_filename = "all_sorts_output_print.txt"

# runs the program
process_files_all_sorts_print(output_filename)

