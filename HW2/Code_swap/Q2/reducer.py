#!/usr/bin/python
import sys

def main():
    current_loc = None
    counter = 0

    for row in sys.stdin:
        key, val = row.strip().split('\t')
        
        if current_loc is None:
            current_loc = key
            counter += 1
        elif key == current_loc:
            counter += 1
        else:
            print('{0}\t{1}'.format(current_loc, counter))
            current_loc = key
            counter = 1
    print('{0}\t{1}'.format(current_loc, counter))

if __name__ == '__main__':
    main()
        

