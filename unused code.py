def split_quarters_manual(matrix):  # split matrix into quarters
    a = matrix
    b = matrix
    c = matrix
    d = matrix
    while len(a) > len(matrix)/2:
        a = a[:len(a)//2]
        b = b[:len(b)//2]
        c = c[len(c)//2:]
        d = d[len(d)//2:]
    while len(a[0]) > len(matrix[0])/2:
        for i in range(len(a[0])//2):
            a[i] = a[i][:len(a[i])//2]
            b[i] = b[i][len(b[i])//2:]
            c[i] = c[i][:len(c[i])//2]
            d[i] = d[i][len(d[i])//2:]
    return a, b, c, d