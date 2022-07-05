def find(num, vect):
    for ind in range(0, len(vect)-1):
        if (num >= vect[ind] and num <= vect[ind+1]):
            return ind