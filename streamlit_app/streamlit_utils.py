def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


def write_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for idiom, parts, full_components in results:
            if idiom:
                file.write(f"Idiom: {idiom}\n")
                file.write(f"Parts: {parts}\n")
                file.write(f"Full Components: {full_components}\n\n")
            else:
                file.write(f"Original Sentence: {parts[0]}\n\n")

def partition_number(n, ranges):
    # Sort the ranges based on the start position
    ranges.sort(key=lambda x: x[0])
    
    partitions = []
    current = 0
    
    # Iterate over each sorted range
    for start, end in ranges:
        # Add the range before the current start position
        if start > current:
            partitions.append((current, start - 1))
        # Add the current range
        partitions.append((start, end))
        current = end + 1
    
    # Add the range after the last end position
    if current < n:
        partitions.append((current, n))
    
    return partitions


def find_all_occurrences(string, substring):
    start = 0
    positions = []
    all_positions = []
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        positions.append(start)
        start += 1  # Move to the next position to find overlapping substrings
    for pos in positions:
        end_pos = pos + len(substring) -1 # Corrected the typo: Use len(substring) instead of _len(substring)
        all_positions.append((pos, end_pos))  # Enclose (pos, end_pos) in a tuple before appending
    return all_positions



def separate_idioms(sentences, idioms):
    print(idioms)
    print(sentences)
    results = []
    for sentence in sentences:
        print(len(sentence))
        found = False
        idiom_positions = []
        for idiom in idioms:
            if idiom in sentence:
                found = True
                for pair in find_all_occurrences(sentence, idiom):
                    idiom_positions.append(pair)
        n =  len(sentence) 
        partions = partition_number(n, idiom_positions)
        idioms = []
        non_idioms = []
        whole_sen = []
        print(partions)
        for pair in partions:
            start = int(pair[0])
            end = int(pair[1])+1
            part = sentence[start:end]
            part = part.strip()
            print(part)

            if pair in idiom_positions:
                idioms.append(part)
                whole_sen.append(part)
            else:
                non_idioms.append(part)
                whole_sen.append(part)
        results.append((idioms, non_idioms, whole_sen))
        if not found:
            results.append((None, [sentence], [sentence]))
        print(results)
    return results