import sys

print "Input the evaluation metrics:"
evaluation = sys.stdin.read()

numbers = []
lines_to_skip = 0
for line in evaluation.split('\n'):
    if not line:
        continue
    if lines_to_skip:
        lines_to_skip -= 1
        continue
    if line.strip().startswith('Raw'):
        lines_to_skip = 4
        continue

    chunks = line.split()
    if chunks[-1] == 'nan':
        continue
    try:
        numbers.append(float(chunks[-1]))
    except ValueError:
        print "Skipping non-numeric line", line

print ' & '.join(["%.1f" % (100 * number) for number in numbers])
