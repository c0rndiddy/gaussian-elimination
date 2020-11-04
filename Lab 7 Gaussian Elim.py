'''
#Needed for array() and dot()
from numpy import *


#Printing matrices using NumPy:

#Convert a list of lists into an array:
M_listoflists = [[1,-2,3],[3,10,1],[1,5,3]]
M = array(M_listoflists)
#Now print it:
print(M)




#Compute M*x for matrix M and vector x by using
#dot. To do that, we need to obtain arrays
#M and x
M = array([[1,-2,3],[3,10,1],[1,5,3]])
x = array([75,10,-11])
b = dot(M,x)

print(M)
#[[ 1 -2  3]
# [ 3 10  1]
# [ 1  5  3]]

#To obtain a list of lists from the array M, we use .tolist()
M_listoflists = M.tolist()

print(M_listoflists) #[[1, -2, 3], [3, 10, 1], [1, 5, 3]]
'''
####
from numpy import *
from numpy import lcm

# Problem 1
def print_matrix(M_lol):
    '''Print nested list M_lol as a matrix
    '''
    for i,j in zip(range(len(M_lol)), range(len(M_lol[0]))):
        M_lol[i][j] = float(M_lol[i][j])
        # this just converts everything in the matrix to a float,
        # not neccesary but kinda aesthetic

    print(array(M_lol))


# Problem 2
def get_lead_ind(row):
    '''Return index of first non-zero element in list row
    if row is all 0s, return length of row
    '''
    for i in range(0, len(row)):
        if row[i] != 0:
            return i

    return len(row)



#Problem 3
def get_row_to_swap(M, start_i):
    '''Check if row start_i in matrix M should be
    swapped with another row below(!) it
    if yes, return the index of the row being swapped
    '''
    left_ind = (get_lead_ind(M[start_i]), start_i)
            # tuple: (index of leftmost integer in row, index of row)

    if left_ind[0] == 0:
        return start_i
        # if first element of row is an integer, row doesn't swap

    else:
        for i in range(start_i, len(M)):
            if get_lead_ind(M[i]) <= get_lead_ind(M[start_i]):
                left_ind = min((get_lead_ind(M[i]), i), left_ind)
                # min chooses the minimum index for the leading integer

        row = left_ind[1]

        return row


# Problem 4
def add_rows_coefs(r1, c1, r2, c2):
    '''Return list obtained by doing scalar multiplication of
    rows r1, r2 by c1, c2 and then adding r1 and r2
    '''
    res = []
    for i in range(len(r1)):
        res.append(r1[i] * c1 + r2[i] * c2)

    return res


# Problem 5
def eliminate(M, row_to_sub, best_lead_ind):
    '''Eliminate the values at best_lead_ind of
    each row after row row_to_sub by subtracing a
    multiple of row row_to_sub from a multiple of them
    '''
    for i in range(row_to_sub+1, len(M)):
        if M[i][best_lead_ind] != 0:
            row_com_mult = lcm(int(M[i][best_lead_ind]), int(M[row_to_sub][best_lead_ind]))
                        # finding the least common multiple between the two rows
            if M[i][best_lead_ind] < 0 or M[row_to_sub][best_lead_ind] < 0:
                row_com_mult = -row_com_mult

            coef1 = row_com_mult / M[i][best_lead_ind]
            coef2 = row_com_mult / M[row_to_sub][best_lead_ind]
            # finding the coefficients to mult each row by to get lcm

            M[i] = add_rows_coefs(M[row_to_sub], -coef2, M[i], coef1)

        else:
            continue

    return M


# Problem 6
def forward_step(M):
    '''Use process of swapping rows and eliminating coefficients
    to convert matrix M to row echelon form
    '''
    for i in range(len(M)):
        print("Now looking at row", i)

        # Swapping rows
        M_copy = M.copy()   # fixing aliasing problem
        M_copy[i] = M[get_row_to_swap(M, i)]
        M_copy[get_row_to_swap(M, i)] = M[i]

        # Will only print the matrix if changes are made
        if M != M_copy:
            print("Swapping rows", i, "and", get_row_to_swap(M, i), "so that entry", get_lead_ind(M[get_row_to_swap(M, i)]), "in the current row is non-zero")

        M = M_copy

        print("The matrix is currently:")
        print_matrix(M)

        # Eliminating coefficients
        M_copy = M_copy.copy()   # another aliasing problem

        # Will only print the matrix if changes are made
        if M_copy != eliminate(M, i, get_lead_ind(M[i])):
            print("Adding row", i, "to rows below it to eliminate coefficients in column", get_lead_ind(M[i]))
            eliminate(M, i, get_lead_ind(M[i]))

            print("The matrix is currently:")
            print_matrix(M)

            print("========================================")

    return M



# Problem 7
def eliminate_above(M, row_to_sub, best_lead_ind):
    '''Same as eliminate but does it for rows above row row_to_sub
    '''
    for i in range(row_to_sub-1, -1, -1):
        if M[i][best_lead_ind] != 0:
            row_com_mult = lcm(int(M[i][best_lead_ind]), int(M[row_to_sub][best_lead_ind]))
                        # finding the least common multiple between the two rows
            if M[i][best_lead_ind] < 0 or M[row_to_sub][best_lead_ind] < 0:
                row_com_mult = -row_com_mult

            coef1 = row_com_mult / M[i][best_lead_ind]
            coef2 = row_com_mult / M[row_to_sub][best_lead_ind]
            # finding the coefficients to mult each row by to get lcm

            M[i] = add_rows_coefs(M[row_to_sub], -coef2, M[i], coef1)

        else:
            continue



def backward_step(M):
    '''Convert matrix M to reduced normal form
    '''
    # Making the column values in every row above the leading integer 0
    for i in range(len(M)-1, -1, -1):
        M_copy = M.copy()
        if M_copy != eliminate_above(M, i, get_lead_ind(M[i])):
            print("Adding row", i, "to rows above it to eliminate coefficients in column", get_lead_ind(M[i]))

            eliminate_above(M, i, get_lead_ind(M[i]))

            print("The matrix is currently:")
            print_matrix(M)

            print("========================================")

    # simplifying the matrix
    print("Now dividing each row by the leading coefficient")
    print("The matrix is currently:")

    M_copy = []     # making a deep copy of M
    for sublist in M:
        M_copy.append(sublist[:])

    for i in range(len(M)):    # dividing each row by its leading coeff
        for j in range(len(M[0])):
            M[i][j] /= (M_copy[i][get_lead_ind(M_copy[i])])

    print_matrix(M)

    print("========================================")

    return M



# Problem 8
'''I don't actually know what problem 8 is about lol help
'''
def solve_that_shit(M):
    print("The matrix is currently:")
    print_matrix(M)

    print("========================================")

    print("Now performing the forward step")
    M = forward_step(M)

    print("========================================")

    print("Now performing the backward step")
    backward_step(M)


if __name__ == '__main__':
    M = [[0, 0, 1, 0, 2],
         [1, 0, 2, 3, 4],
         [3, 0, 4, 2, 1],
         [1, 0, 1, 1, 2]]    # applying forward_step makes this last row [0, 0, 0, 0, 14],
                             # which is different from guerzhoy's example of [0, 0, 0, 0, 2],
                             # but they're scalar multiples of each other, ie the same thing,
                             # so i left it alone, but u can fix it if you want
                             # note that the matrix is eventually simplified
                             # at the end of everything in the backward_step function

    N = [[5, 6, 7, 8],
         [0, 0, 1, 1],
         [0, 0, 0, 2],
         [0, 0, 7, 0]]

    P = [[1, -2, 3, 22],
         [3, 10, 1, 314],
         [1, 5, 3, 92]]

    Q = [[1, -2, 3, 22],
         [0, 16, -8, 248],
         [0, 0, 56, -616]]

    # print(get_lead_ind(M[1])
    # print(get_row_to_swap(M, 1))
    # print(add_rows_coefs(M[0], 2, M[2], 5))
    # print(array(eliminate(Mr2, 2, 3)))

    # forward_step(P)
    # backward_step(P)

    solve_that_shit(P)