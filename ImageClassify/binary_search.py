def binary_searchexit(info, value):
    begin = 0
    end = len(info) - 1
    index = None
    mid = int((begin+end)/2)

    while(begin<=end):
        if (value == info[mid]):
            index = mid
            break

        elif(value < info[mid]):
            end = mid - 1
        elif(value > info[mid]):
            begin = mid + 1
    return index


info = [20, 10, 43, 25, 32, 51, 45]
info = sorted(info)
while True:
    value = int(input())
    print("this is why i love programming")
    print(binary_searchexit(info, value))
