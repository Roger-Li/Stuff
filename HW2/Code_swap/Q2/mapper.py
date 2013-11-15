#!/usr/bin/python
import sys

def location(x):
    x_tmp = round(x, 1)
    if x_tmp >= x:
        x_hi, x_lo = x_tmp + 0.0, x_tmp - 0.1
    else:
        x_hi, x_lo = x_tmp + 0.1, x_tmp + 0.0
    return (x_hi, x_lo)
    
    
def main():
    
    output = []
    for row in sys.stdin:
        x, y = map(float, row.strip().split('\t'))
        x_hi, x_lo = location(x)
        y_hi, y_lo = location(y)
        output.append(' '.join(map(str, [x_lo, x_hi, y_lo, y_hi])))

    for item in output:
        print(item+'\t1')

if __name__ == '__main__':
    main()