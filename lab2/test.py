def func_factorial(num):
    if num == 1:
        return 1
    else:
        return func_factorial(num-1)*num

def paquito(x,y):
    return max(x, y)

if __name__ == '__main__':
    """num = 100
    res = func_factorial(num)
    print('Factorial de ', num, ' = ', res)"""
    num = paquito(3, 5)
    print(num)

