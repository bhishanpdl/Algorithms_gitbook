<a id="top"></a>


Table of Contents
=================
   * [1 LC 1365: How Many Numbers Are Smaller Than the Current Number](#1-lc-1365-how-many-numbers-are-smaller-than-the-current-number)
   * [2 LC 1313: Decompress Run-Length Encoded List](#2-lc-1313-decompress-run-length-encoded-list)
   * [3 LC 1295: Find Numbers with Even Number of Digits](#3-lc-1295-find-numbers-with-even-number-of-digits)
   * [4 LC 1266: Minimum Time Visiting All Points](#4-lc-1266-minimum-time-visiting-all-points)
   * [5 LC 1351: Count Negative Numbers in a Sorted Matrix](#5-lc-1351-count-negative-numbers-in-a-sorted-matrix)
   * [6 LC 1252: Cells with Odd Values in a Matrix](#6-lc-1252-cells-with-odd-values-in-a-matrix)
   * [7 LC 1299: Replace Elements with Greatest Element on Right Side](#7-lc-1299-replace-elements-with-greatest-element-on-right-side)
   * [8 LC 1304: Find N Unique Integers Sum up to Zero](#8-lc-1304-find-n-unique-integers-sum-up-to-zero)
   * [9 LC 832: Flipping an Image](#9-lc-832-flipping-an-image)
   * [10 LC 905: Sort Array By Parity](#10-lc-905-sort-array-by-parity)
   * [11 LC 977: Squares of a Sorted Array](#11-lc-977-squares-of-a-sorted-array)
   * [12 LC 561: Array Partition I](#12-lc-561-array-partition-i)
   * [13 LC 1051: Height Checker](#13-lc-1051-height-checker)
   * [14 LC 1337: The K Weakest Rows in a Matrix](#14-lc-1337-the-k-weakest-rows-in-a-matrix)
   * [15 LC 922: Sort Array By Parity II](#15-lc-922-sort-array-by-parity-ii)
   * [16 LC 1122: Relative Sort Array](#16-lc-1122-relative-sort-array)
   * [17 LC 1160:  Find Words That Can Be Formed by Characters](#17-lc-1160--find-words-that-can-be-formed-by-characters)
   * [18 LC 70: Fibonacci and 509: Climbing Stairs](#18-lc-70-fibonacci-and-509-climbing-stairs)
   * [19 LC 1002:  Find Common Characters](#19-lc-1002--find-common-characters)
   * [20 LC 1200:  Minimum Absolute Difference](#20-lc-1200--minimum-absolute-difference)
   * [21 LC 999:  Available Captures for Rook](#21-lc-999--available-captures-for-rook)
   * [22 LC 1185:  Day of the Week](#22-lc-1185--day-of-the-week)
   * [23 LC 1217:  Play with Chips](#23-lc-1217--play-with-chips)
   * [24 LC 766:  Toeplitz Matrix](#24-lc-766--toeplitz-matrix)
   * [25 LC 867:  Transpose Matrix](#25-lc-867--transpose-matrix)
   * [26 LC 985:  Sum of Even Numbers After Queries](#26-lc-985--sum-of-even-numbers-after-queries)
   * [XX LC 1: Two Sum](#xx-lc-1-two-sum)
   * [XX LC 784: Letter Case Permutation](#xx-lc-784-letter-case-permutation)

# 1 LC 1365: How Many Numbers Are Smaller Than the Current Number

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>

- [how-many-numbers-are-smaller-than-the-current-number](https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/)
![](../images/lc_1365.png)
```python
nums = [8,1,2,2,3]
[sorted(nums).index(n) for n in nums]
```

# 2 LC 1313: Decompress Run-Length Encoded List

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>

- [decompress-run-length-encoded-list](https://leetcode.com/problems/decompress-run-length-encoded-list/)
![](../images/lc_1313.png)
```python
nums = [1,2,3,4]

#======================================== easy
ans = [] # O(n) very good
for i in range(len(nums)): # O(n)
    if i%2: # odd index  O(1)
        ans += [ nums[i] ] * nums[i-1]
#========================================
[ vf for v,f in zip(nums[1::2],nums[0::2]) for vf in [v]*f ]    # O(n/2)*O(n/2)
[nums[i+1] for i in range(0, len(nums), 2) for _ in range(nums[i])] # O(n/2) * O(n)
sum(([nums[i+1]]*nums[i] for i in range(0, len(nums), 2)), [])  # O(n) + O(n)

two steps:
ans = [[val]*freq for val,freq in zip(nums[1::2],nums[0::2])]
ans = [i for sublist in ans for i in sublist]

expansion: This is O(n**2)
ans = []
for v,f in zip(nums[1::2],nums[0::2]): # O(n/2) is O(n)
    for vf in [v]*f: # O(f) is O(n) here [4]*3 = [4,4,4]
        ans.append(vf)
ans
```

# 3 LC 1295: Find Numbers with Even Number of Digits

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [find-numbers-with-even-number-of-digits](https://leetcode.com/problems/find-numbers-with-even-number-of-digits/)
![](../images/lc_1295.png)
```python
nums = [12,345,2,6,7896]
ans =  len([i for i in nums if len(str(i))%2 == 0]) # needs to create list
ans = sum(len(str(num)) % 2 == 0 for num in nums)

len((i for i in nums if len(str(i))%2 == 0))
# TypeError: object of type 'generator' has no len()

ans = 0
for num in nums:
    if len(str(num))%2 == 0: # aliter: if not len(str(num))%2:
        ans+=1
```

# 4 LC 1266: Minimum Time Visiting All Points

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [minimum-time-visiting-all-points](https://leetcode.com/problems/minimum-time-visiting-all-points/)
![](../images/lc_1266.png)
```python
class Solution:
    def minTimeToVisitAllPoints(self, points):
        dist = 0
        for i in range(len(points) - 1):
            dx = abs(points[i+1][0] - points[i][0])
            dy = abs(points[i+1][1] - points[i][1])
            dist += max(dx, dy)
            print(f"i={i}, dx={dx},dy={dy},dist={dist}")

        return dist

points = [[1,1],[3,4],[-1,0]]
sol = Solution()
sol.minTimeToVisitAllPoints(points) # 7
i=0, dx=2,dy=3,dist=3
i=1, dx=4,dy=4,dist=7

We need to move at least the longest distance between
corresponding coordinates in order to get to the point
for example:
[-111, 222] and [0, 15]
abs(-111-0) = 111
abs(222 - 15) = 207
Then we need to move 207 to get to [0, 15]
```

# 5 LC 1351: Count Negative Numbers in a Sorted Matrix

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [count-negative-numbers-in-a-sorted-matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/)
![](../images/lc_1351.png)
```python

#========================================= m * nlogn
class Solution:
    def count_negatives(self, grid):
        def negs_in_row(row):
            s,e = 0, len(row) # row = [4,3,2,-1] s=0 e=4
            while s < e: # to end while loop we need to increase s or decrease e
                mid = s + (e-s)//2 # mid = 0 + (4-0)//2 = 0 + 2 = 2 (middle index)
                if row[mid]<0:     # middle element row[mid]=row[2]=2
                    e = mid        # if middle element is negative, make it end
                else:              # else, again start from middle
                    s = mid + 1    # NOTE: end is mid but start is mid +1
            return len(row) - s

        cnt = 0
        for row in grid:
            cnt += negs_in_row(row)

        return cnt

Explanation:
def negs_in_row(row):
    s,e = 0,len(row)
    print(row)
    while s < e:
        mid = s + (e-s)//2
        if row[mid]<0:
            print(f'if  : s={s} e={e} mid={mid} row[mid]={row[mid]} '+
                  f' Negative, e decreases to mid={mid} and s is same.')
            e = mid
        else:
            print(f'else: s={s} e={e} mid={mid} row[mid]={row[mid]} '+
                  f'  NOT Negative, s increases to mid+1={mid + 1}')
            s = mid + 1
    return len(row) - s

grid = [[4,3,-2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
negs_in_row(grid[0])
[4, 3, -2, -1]
if  : s=0 e=4 mid=2 row[mid]=-2  Negative, e decreases to mid=2 and s is same.
else: s=0 e=2 mid=1 row[mid]=3   NOT Negative, s increases to mid+1=2
2

sol = Solution()
sol.countNegatives(grid)

#========================================= < O(m*n)
class Solution:
    def count_negatives(self, grid):
        h, w = len(grid), len(grid[0])
        cnt_neg = 0
        for y in range(h):
            for x in range(w):
                if grid[y][x] < 0:
                    print(y,x)
                    cnt_neg += (w-x)
                    break # break the row if seen -ve

        return cnt_neg
#========================================== O(mn)
sum( 1 for row in grid for element in row if element < 0 )
```

# 6 LC 1252: Cells with Odd Values in a Matrix

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [cells-with-odd-values-in-a-matrix](https://leetcode.com/problems/cells-with-odd-values-in-a-matrix/)
![](../images/lc_1252.png)
```python
"""
            ri ci           * index 0,1 increase row and column by 1 (note: index01 becomes 2)
                                 row0   col1    row1  col1
                                 *                *
indices[0]  0  1          0 0 0* 1 1 1  1 2 1   1 2 1 1 3 1
indices[1]  1  1          0 0 0  0 0 0  0 1 0*  1 2 1 1 3 1
"""
# NOTE: Leetcode needs function name oddCells instead of odd_cells
# but I have changed n,m to nr,nc as number of rows and columns
class Solution:
    def oddCells(self, nr, nc, indices):
        # Array
        A = [[0 for _ in range(nc)] for _ in range(nr)]
        for index_row in indices:
            i,j = index_row # tuple unpacking
            for _ in range(nc): A[i][_] += 1
            for _ in range(nr): A[_][j] += 1

        num_odds = sum(1 for row in A for elem in row if elem%2 != 0)
        # here we don't need list comprehension, we use only iterator
        return num_odds

indices = [[0,1],
           [1,1]
           ]
sol = Solution()
sol.oddCells(2,3,indices)
```

# 7 LC 1299: Replace Elements with Greatest Element on Right Side

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [replace-elements-with-greatest-element-on-right-side](https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/submissions/)
![](../images/lc_1299.png)
```python
class Solution:
    def replaceElements(self, arr):
        L = len(arr)
        rt_max = -1

        # reversed for loop and swap elements (inplace operation)
        for i in range(L-1, -1, -1):
            arr[i], rt_max = rt_max, max(rt_max, arr[i])
        return arr

Simplified version of above:
class Solution:
    def replaceElements(self, arr):
        L = len(arr)
        rt_max = -1

        # reversed for loop and change rt_max if larger than hold
        # We start with L-1 since arr[L] gives index error
        for i in range(L-1, -1, -1):
            hold = arr[i]   # current element
            arr[i] = rt_max # change arr[i] to rt_max (last becomes -1)
            if hold > rt_max:
                rt_max = hold
        return arr

for arr in [ [], None, [17,18,5,4,6,1]]:
    sol = Solution()
    out = sol.replaceElements(arr) # [18, 6, 6, 6, 1, -1]
    print(out)
```

# 8 LC 1304: Find N Unique Integers Sum up to Zero

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [find-n-unique-integers-sum-up-to-zero](https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/)
![](../images/lc_1304.png)
```python
Symmetric solution stride of 2
===================================
class Solution:
    def sumZero(self, n):
        return list(range(1 - n, n, 2))
n = 5
sol = Solution()
sol.sumZero(n) # [-4, -2, 0, 2, 4]

Symmetric Solution stride of 1
===============================
class Solution:
    def sumZero(self, n):
        return (list(range(-(n//2), 0))
                + [0]*(n % 2)
                + list(range(1, n//2 + 1)))
n = 5
sol = Solution()
sol.sumZero(n) # [-2, -1, 0, 1, 2]

Last Element is sum of others
===============================
class Solution:
    def sumZero(self, n):
        return list(range(1,n))+[-n*(n-1)//2]
n = 4
sol = Solution()
sol.sumZero(n) # [1, 2, 3, -10]  check: 1+2+3 = 6  4*3//2 = 6
```

# 9 LC 832: Flipping an Image

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [flipping-an-image](https://leetcode.com/problems/flipping-an-image/)
![](../images/lc_832.png)
```python
# Using reversed
[[ 0 if i else 1 for i in reversed(row)] for row in A]

# Using [::-1]
[[ 0 if i else 1 for i in row[::-1]] for row in A]

# Using XOR
[[ i ^ 1 for i in reversed(row)] for row in A]

# Using int not
[[ int(not i) for i in reversed(row)] for row in A]

# Using subtraction
[[ 1 - i for i in reversed(row)] for row in A]

# Without list comp
out = []
for row in A:
    flipped = reversed(row) # row[::-1] # flipped means reversed
    inverted = [i^1 for i in flipped] # inverted means xor operation ^
    out.append(inverted)
#====================================== using numpy
import numpy as np

A = \
[[1 1 0]
 [1 0 1]
 [0 0 0]]

np.fliplr(A)
[[0 1 1]
 [1 0 1]
 [0 0 0]]

 1 - np.fliplr(A) # np.bitwise_xor(np.fliplr(A),1)
 [[1 0 0]
 [0 1 0]
 [1 1 1]]
```

# 10 LC 905: Sort Array By Parity

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [sort-array-by-parity](https://leetcode.com/problems/sort-array-by-parity/)
![](../images/lc_905.png)
```python
#========================== Simple while loop

class Solution:
        def sortArrayByParity(self, A) :
            if not A: return []
            l,r = 0, len(A) - 1
            while(r >= l): # while gt equal to

                # bad: left odd, right even: swap and update both ptr
                if A[l] % 2 == 1 and A[r] % 2 == 0:
                    A[r], A[l] = A[l], A[r]
                    l +=1
                    r -=1

                # good: left even, increase left ptr
                if A[l] % 2 == 0: l += 1

                # good: right odd, decrease right ptr
                if A[r] % 2 == 1: r -= 1
            return A

A = [3,1,2,4]
sol = Solution()
sol.sortArrayByParity(A)
#============================= Builtin and list comp
ans = sorted(A,key=lambda x: x%2) # NOTE: i%2 means ODD
ans = [i for i in A if not i % 2] + [i for i in A if i % 2]
ans = [i for i in A if i % 2 == 0] + [i for i in A if i % 2 !=0]
```

# 11 LC 977: Squares of a Sorted Array

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [squares-of-a-sorted-array](https://leetcode.com/problems/squares-of-a-sorted-array/)
- [squares-of-a-sorted-array](https://leetcode.com/problems/squares-of-a-sorted-array/discuss/310865/Python%3A-A-comparison-of-lots-of-approaches!-Sorting-two-pointers-deque-iterator-generator)
- [medium: squares-of-a-sorted-array](https://medium.com/@george.seif94/)this-is-the-fastest-sorting-algorithm-ever-b5cee86b559c
![](../images/lc_977.png)

```python
#========================================== Two pointers reversed for loop
class Solution:
    def sortedSquares(self, A) :
        if not A:
            return []
        L = len(A)
        res = [0 for _ in range(L) ]
        l, r = 0, len(A) - 1

        # reversed iteration
        for i in range(L-1, -1, -1):
            if abs(A[l]) > abs(A[r]):
                res[i] = A[l] ** 2
                l += 1
            else:
                res[i] = A[r] ** 2
                r -= 1
        return res

#============================== DP + Two pointers while from end
class Solution:
    def sortedSquares(self, A) :

        # edge case
        if not A: return -1

        # length
        L = len(A)

        # init result
        res = [0 for _ in range(L) ]

        # two pointers left and right
        l = 0
        r = L - 1

        # left and right square
        lsq = A[l] ** 2
        rsq = A[r] ** 2

        # pointer to write from last
        i = L - 1
        while i >= 0:
            if lsq > rsq:
                res[i] = lsq
                l += 1
                lsq = A[l] ** 2
            else:
                res[i] = rsq
                r -= 1
                rsq = A[r] ** 2
            i -= 1
        return res

for A in [[-2,-1,0,0,2,3], [], None, [-5], [-7,-3,2,3,11] ]:
    sol = Solution()
    print(A, sol.sortedSquares(A))


#========================================== Inplace
class Solution:
    def sortedSquares(self, A) :
        res = [v**2 for v in A]
        res.sort() # inplace operation
        return res

#========================================== Built-in sorted (Timsort O(n) not nlogn) Fastest
class Solution:
    def sortedSquares(self, A) :
            return sorted([v**2 for v in A])

#========================================== Using collections.deque
import collections

class Solution:
    def sortedSquares(self, A):
        dq = collections.deque()
        l, r = 0, len(A) - 1
        while l <= r:
            left, right = abs(A[l]), abs(A[r])
            if left > right:
                dq.appendleft(left * left)
                l += 1
            else:
                dq.appendleft(right * right)
                r -= 1
        return list(dq)
```

# 12 LC 561: Array Partition I

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [array-partition-i](https://leetcode.com/problems/array-partition-i/)
![](../images/lc_561.png)
```python
class Solution:
    def arrayPairSum(self, nums):
        return sum(sorted(nums)[::2])
```

# 13 LC 1051: Height Checker

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- https://leetcode.com/problems/height-checker/
- https://www.youtube.com/watch?v=TTnvXY82dtM
![](../images/lc_1051.png)
```python
#======================= Using counting sort
** Counting sort is stable.
** It is good when range of numbers is small. e.g Students with similar heights
** It works for only +ve numbers but can be extended to -ve numbers.
** For large numbers, we can use Radix sort, which is built upon Count sort.
** Time Complexity: O(n+k) k is range of numbers 0 to maxNum
** Auxiliary Space: O(n+k)

class Solution:
    def heightChecker(self, heights):
        max_ht = max(heights)

        # freq table
        freq = [0] * (max_ht + 1)
        for h in heights: freq[h] += 1
        for i in range(1, len(freq)): freq[i] += freq[i-1]

        # places for heights
        places = [0] * len(heights)
        for h in heights:
            places[freq[h]-1] = h
            freq[h] -= 1

        return sum([h!=p for h, p in zip(heights, places)])

# method 2
sum([h1!=h2 for h1,h2 in zip(sorted(heights),heights)])
sum((h1!=h2 for h1,h2 in zip(sorted(heights),heights))) # iterator version
sum(map(operator.ne, heights, sorted(heights))) # import iterator
h1=np.array(heights); h2=np.sort(h1); ans=(h1!=h2).sum() # numpy
```

# 14 LC 1337: The K Weakest Rows in a Matrix

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [the-k-weakest-rows-in-a-matrix](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/)
![](../images/lc_1337.png)
```python
class Solution:
    def kWeakestRows(self, A, k) : # A is an array
        S = [[sum(row),i] for i,row in enumerate(A)] # rowsum, index
        S = sorted(S)
        return [row[1] for row in S[:k]]

# method 2
[row[1] for row in heapq.nsmallest(k,[[sum(a),i]
     for i,a in enumerate(A)])]

# method 3
sorted(range(len(mat)), key=mat.__getitem__)[:k]
sorted(range(len(mat)), key=lambda row: sum(mat[row]))[:k]

# explanation
mat = \
[[1,1,0,0,0],
 [1,1,1,1,0],
 [1,0,0,0,0],
 [1,1,0,0,0],
 [1,1,1,1,1]]

len(mat) = 5 there are 5 rows
list(range(len(mat))) = [0,1,2,3,4]
m0 = mat.__getitem__(0) # [1, 1, 0, 0, 0] first row
m1 = mat.__getitem__(1) # [1, 1, 1, 1, 0] second row
m0 < m1 True   we can use getitem attribute as comparator.

Calculating the sum of a row has a time complexity of O(n).
Sorting has a complexity of O(m log m).
The complexities get multiplied because the row sums are not cached.

Time complexity: O((m log m) * n)
Space complexity: O(m)
```

# 15 LC 922: Sort Array By Parity II

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [sort-array-by-parity-ii](https://leetcode.com/problems/sort-array-by-parity-ii/)
![](../images/lc_922.png)
```python
# method1: inplace change using while odd less than Length
class Solution:
    def sortArrayByParityII(self, A):
        i, j, L = 0, 1, len(A) # i is even j is odd
        while j < L:
            if A[j] % 2 == 0:
                A[j], A[i] = A[i], A[j]
                i += 2
            else:
                j += 2
        return A

# method 2: using new array and put even/odd continuously
class Solution:
    def sortArrayByParityII(self, A):
        # result
        res = [0]*len(A)

        # two pointers for even and odd
        i, j = 0, 1

        # create new array and put element from
        # if a is even put to even and increase by 2 and ditto
        for a in A:
            if a % 2 == 0: res[i] = a; i += 2
            else         : res[j] = a; j += 2
        return res

A = [4, 2, 5, 7]
sol = Solution()
sol.sortArrayByParityII(A)

concise version
class Solution:
    def sortArrayByParityII(self, A):
        index = [0, 1] #even & odd indices
        sorted_arr = [0] * len(A)
        for a in A: # [4,2,5,7]
            sorted_arr[index[a%2]] = a
            # 4%2 = 0 sorted_arr[0] becomes A[0]=4 and index 0 becomes 2
            # 2%2 = 0 sorted_arr[0+2] becomes A[1]=2 and index 0 becomes 4
            index[a%2] += 2
        return sorted_arr
```

# 16 LC 1122: Relative Sort Array

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [relative-sort-array](https://leetcode.com/problems/relative-sort-array/)
![](../images/lc_1122.png)
```python
#======================= Counter and Counter.pop
class Solution:
    def relativeSortArray(self, arr1, arr2):
        c = collections.Counter(arr1)
        res = []
        for a2 in arr2:
            res += [a2]*c.pop(a2)
        return res + sorted(c.elements())

arr1 = [2,3,1,3,2,4,6,7,9,2,19]
arr2 = [2,1,4,3,9,6]
sol = Solution()
sol.relativeSortArray(arr1,arr2)
# c ==> Counter({2: 3, 3: 2, 1: 1, 4: 1, 6: 1, 7: 1, 9: 1, 19: 1})
# NOTE: Counter is already sorted by frequency
# c.pop(2) gives 3 then,
# c becomes Counter({3: 2, 1: 1, 4: 1, 6: 1, 7: 1, 9: 1, 19: 1})

#====================== List comp (very slow)
class Solution:
    def relativeSortArray(self, arr1, arr2):
        a = [x for i in range(len(arr2)) for x in arr1 if x == arr2[i]]
        b = sorted([x for x in arr1 if x not in arr2])
        return a + b

Explanation:
for i in range(len(arr2)):
    for x in arr1:
        if x == arr2[i]:
            print(x,end=' ')
[2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]

#====================== functools cmp_to_key (slowest)
from functools import cmp_to_key
class Solution:
    def relativeSortArray(self, arr1, arr2):
        def cmp(x,y):
            if x not in arr2 and y in arr2:
                return 1
            elif x in arr2 and y not in arr2:
                return -1
            elif x not in arr2 and y not in arr2:
                return  1 if x>y else -1
            elif x in arr2 and y in arr2:
                return 1 if arr2.index(x) > arr2.index(y) else -1

        return sorted(arr1, key = cmp_to_key(cmp))
```

# 17 LC 1160:  Find Words That Can Be Formed by Characters

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [find-words-that-can-be-formed-by-characters](https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/)
![](../images/lc_1160.png)
```python
words = ["hello","world","leetcode"]
chars = "welldonehoneyr"

class Solution:
    def countCharacters(self, words, chars):
        ans = 0
        for word in words:
            flag = True # initially assume all words are there inside for loop
            for charw in word: # if word has more characters break loop with flag false
                if word.count(charw)>chars.count(charw):
                    flag = False
                    break

            # outside of for-loop
            if flag:
                ans += len(word)
        return ans
sol = Solution()
sol.countCharacters(words,chars)
#=========================== Using Counter (one-liner)
from collections import Counter
class Solution:
    def countCharacters(self, words, chars):
        return sum(not Counter(w) - Counter(chars) and len(w) for w in words)
#=========================== Using Counter and flags
from collections import Counter
class Solution:
    def countCharacters(self, words, chars):
	# count chars character counts
        chctr = Counter(chars)
        goodln = 0
        for word in words:
            wctr = Counter(word)
            isgood = True
            for key in wctr:
		        # chech if the character (or key) is available &
		        # we have enough of it
                if key not in chctr or wctr[key] > chctr[key]:
                    isgood = False
                    break
            if isgood:
                goodln += len(word)
        return goodln
```
# 18 LC 70: Fibonacci and 509: Climbing Stairs

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


 - [509: Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
 - [70: Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

- https://en.wikipedia.org/wiki/Fibonacci_number#Closed-form_expression
- https://leetcode.com/problems/climbing-stairs/discuss/25313/Python-different-solutions-(bottom-up-top-down).

In mathematics, the Fibonacci numbers, commonly denoted Fn, form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

${\displaystyle F_{0}=0,\quad F_{1}=1}$
and

${\displaystyle F_{n}=F_{n-1}+F_{n-2}}$
for n > 1.

The beginning of the sequence is thus:
```
0 1 2 3 4 5 6 7  8  9  10 11 12
0 1 1 2 3 5 8 13 21 34 55 89 144
```


Closed form solution
$$
a_{n}=1 / \sqrt{5}\left[\left(\frac{1+\sqrt{5}}{2}\right)^{n}-\left(\frac{1-\sqrt{5}}{2}\right)^{n}\right]
$$

```python
#=========================================== DP Bottom up
# DP 24 ms
class Solution:
    def fib(self, n):
        # edge case
        if n == 0:
            return 1

        # init dp
        dp = [0] * (n+1) # we also need 0

        # base cases
        dp[0],dp[1] = 1,1

        # fill dp
        for i in range(2,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]

for n in [0,1,2,3,4,5,6,7,8]:
    sol = Solution()
    print(f'F({n}) = {sol.fib(n)}')

# 24 sec
# Top down + memoization (dictionary)
#======================================= DP Top Down Memoization Dictionary (fill dictionary recursively)
class Solution:
    # fibonacci for n>1: F(n) = F(n-1) + F(n-2)
    def __init__(self):
        self.dic = {1:1, 2:1}

    def fib(self, n):
        # edge case
        if n == 0: return 0

        # when n > 1
        if n not in self.dic:
            self.dic[n] = self.fib(n-1) + self.fib(n-2)
        return self.dic[n]

#=========================================== Direct Formula
# 24ms
class Solution:
    def fib(self, n):
        return int((5**.5 / 5) * (((1 + 5**.5)/2)**(n + 1) - ((1 - 5**.5)/2)**(n + 1)))

#=========================================== Recursion
# Time limit exceeded,
class Solution:
    # Top down - TLE
    def fib(self, n):
        if n == 1:
            return 1
        if n == 2:
            return 2
        return self.fib(n-1)+self.fib(n-2)

#=========================================== DP Bottom up
# O(n) space
class Solution:
    def fib(self, n):
        # edge case
        if n == 0:
            return 0

        # init dp
        dp = [0] * (n+1)

        # base cases
        dp[0], dp[1] = 1, 1

        # fill dp
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]

# 52 ms
# Bottom up, constant space
#======================================== Two pointers
class Solution:
    def fib(self, n):
        # edge case
        if n == 0:
            return 0

	    # two pointers
        a, b = 1, 1 # F1=1 F2=1 we start with 2
        for i in range(2, n+1): # range n+1 will include n
            a,b = b,a+b
        return b

#=========================================== DP Top Down, Memoization List
# 28 ms
# Top down + memoization (list)
class Solution:
    def fib(self, n):
        # edge case
        if n == 1:
            return 1

        # init dp
        dp =  [-1] * n

        # base case
        dp[0], dp[1] = 1, 2

        # run helper function
        return self._helper(n-1, dp)

        # this function runs recursively and goes to smallest n and finds dp[i] and dp[i-1] there.
    def _helper(self, n, dp):
        if dp[n] < 0: # make sure dp is initialized with -1 then only we can use this.
            dp[n] = self._helper(n-1, dp)+self._helper(n-2, dp)
        return dp[n]

# 32ms
# lru cache
#========================================= Recursion with LRU Cache
from functools import lru_cache

class Solution:
    @lru_cache(None)
    def fib(self, n):
        # check errors
        assert isinstance(n,int), 'n must be an integer'
        assert n >= 0, 'n must be>=0'

        # base cases
        if n == 0: return 0
        if n == 1: return 1

        # recursion
        else:
            return self.fib(n-1) + self.fib(n-2)
```

# 19 LC 1002: Find Common Characters

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [find-words-that-can-be-formed-by-characters](https://leetcode.com/problems/find-common-characters/)
![](../images/lc_1002.png)
```python
A = ['bella','label','roller']

class Solution:
    def commonChars(self, A):
        # edge case
        if len(A)<2 : return A
        out = []
        for c in set(A[0]): # if A is empty A[0] fails
            n = min([a.count(c) for a in A])
            if n:
                out += [c]*n
        return out

# making one-liner
[c for c in set(A[0]) for _ in
  range(min(a.count(c) for a in A))] # fails for empty list

[ (c for c in set(A[0]))
 if len(A) > 0
 else A
 for _ in range(min([a.count(c) for a in A]+[0]))
]

#================= Using counter
from collections import Counter
from functools import reduce
class Solution:
    def commonChars(self, A):
        if len(A) < 2: return A
        ctr = Counter(A[0])
        for a in A[1:]: ctr &= Counter(a)
        return list(ctr.elements())

# one liner
list(reduce(collections.Counter.__and__, map(collections.Counter, A)).elements()

As = [[],['a'],['bella','label','roller'],
["cool","lock","cook"]]
for A in As:
    sol = Solution()
    print(sol.commonChars(A))
```

# 20 LC 1200:  Minimum Absolute Difference

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [minimum-absolute-difference](https://leetcode.com/problems/minimum-absolute-difference/)
![](../images/lc_1200.png)
```python
class Solution:
    def minimumAbsDifference(self, arr):
        arr.sort()
        mn = min(b - a for a, b in zip(arr, arr[1:]))
        return [[a, b] for a, b in zip(arr, arr[1:]) if b - a == mn]

#==================== simple
class Solution:
    def minimumAbsDifference(self,arr):
        # edge case
        if len(arr) < 2: return arr
        s = sorted(arr) # for inplace use arr.sort()
        #ds = [s[i]-s[i-1] for i in range(1,len(s))]
        #minds = min(ds)
        minds = min(s[i]-s[i-1] for i in range(1,len(s)))
        out = []
        for i in range(1,len(s)):
            if s[i] - s[i-1] == minds:
                o = [s[i-1],s[i]]
                if o not in out:
                    out.append(o)
        return out

arr1 = [4,2,1,3]
arr2 = [1,3,6,10,15]
arr3 = [3,8,-10,23,19,-4,-14,27]
sol = Solution()
for arr in [arr1,arr2,arr3]:
    print(sol.minimumAbsDifference(arr))
```

# 21 LC 999:  Available Captures for Rook

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [available-captures-for-rook](https://leetcode.com/problems/available-captures-for-rook/)
![](../images/lc_999a.png)
![](../images/lc_999b.png)
![](../images/lc_999c.png)
![](../images/lc_999d.png)
```python
class Solution:
    def numRookCaptures(self, B):
        y, x = next((i, j) for j in range(8) for i in range(8) if B[i][j] == 'R')
        row = ''.join(B[y][j] for j in range(8) if B[y][j] != '.')
        col = ''.join(B[i][x] for i in range(8) if B[i][x] != '.')
        return sum('Rp' in s for s in (row, col)) + sum('pR' in s for s in (row, col))

B = [[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","p",".",".",".","."],["p","p",".","R",".","p","B","."],[".",".",".",".",".",".",".","."],[".",".",".","B",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."]]
sol = Solution(); sol.numRookCaptures(B)

#======================
class Solution:
    def numRookCaptures(self, board):
        # find 'R'
        for row in range(8):
            for col in range(8):
                if board[row][col] == 'R':
                    # build two strings for the row and the column
                    r = ''.join(x for x in board[row] if x != '.')
                    c = ''.join(board[i][col] for i in range(8) if board[i][col] != '.')
                    # count the number of 'Rp' substring
                    return sum('Rp' in x for x in (r, r[::-1], c, c[::-1]))


# plain simple way
class Solution:
    def numRookCaptures(self, B):
        # find position of rook
        x=0
        y=0
        for i in range(8):
            for j in range(8):
                if B[i][j] == 'R':
                    x = i
                    y = j

        # move 4 directions
        ans = 0

        # move +ve x-axis (row is fixed, y increases)
        for i in range(y+1,8):
            e = B[x][i]
            if e == '.': continue
            elif e == 'p': ans +=1; break
            else: break

        # move -ve x-axis
        for i in range(y-1,-1,-1):
            e = B[x][i]
            if e == '.': continue
            elif e == 'p': ans +=1; break
            else: break

        # move +ve y-axis
        for i in range(x+1,8):
            e = B[i][y]
            if e == '.': continue
            elif e == 'p': ans +=1; break
            else: break

        # move -ve y-axis
        for i in range(x-1,-1,-1):
            e = B[i][y]
            if e == '.': continue
            elif e == 'p': ans +=1; break
            else: break

        return ans
```

# 22 LC 1185:  Day of the Week

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [day-of-the-week](https://leetcode.com/problems/day-of-the-week/)
![](../images/lc_1185.png)
```python
import datetime, calendar

d,m,y = (31,8,2019)
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] # same as calendar.day_name
datetime.date(y,m,d).strftime("%A") # 'Saturday'
dateutil.parser.parse(f"{y}-{m}-{d}").strftime("%A")
datetime.date(y,m,d).weekday() # gives 5, then do days[5] = Saturday
pd.Timestamp(f"{y}-{m}-{d}").day_name() # aliter: .strftime("%A")
pd.DatetimeIndex([f"{y}-{m}-{d}"]).day_name()[0]
```

# 23 LC 1217:  Play with Chips

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [play-with-chips](https://leetcode.com/problems/play-with-chips/)
![](../images/lc_1217.png)
```python
NOTE: All the chips must be stacked at same place.
Logic: each element can be moved to location 0 or location 1 without cost cause moving 2 steps is free.
class Solution:
    def minCostToMoveChips(self, chips):
        odds = 0
        for chip in chips:
            if chip % 2 == 1:
                odds += 1

        return min(odds,len(chips)-odds)

#=========================== Using Counter
chips = [2,2,2,3,3]
ctr = collections.Counter([c % 2 for c in chips]) # Counter({0: 3, 1: 2}
ans = min(ctr[0], ctr[1])
sol = Solution(); sol.minCostToMoveChips(chips)
```

# 24 LC 766:  Toeplitz Matrix

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [toeplitz-matrix](https://leetcode.com/problems/toeplitz-matrix/)
![](../images/lc_766.png)
```python
A = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
class Solution:
    def isToeplitzMatrix(self, A):
        if not A: return False
        for i in range(1, len(A)):
            for j in range(1, len(A[0])):
                if A[i][j] != A[i -1][j - 1]:
                    return False
        return True

class Solution:
    def isToeplitzMatrix(self, A):
        for i in range(len(A) - 1):
            for j in range(len(A[0]) - 1):
                if A[i][j] != A[i + 1][j + 1]:
                    return False
        return True

class Solution:
    def isToeplitzMatrix(self, A):
        for i in range(len(A)-1): # compare shifted row
            if A[i][:-1] != A[i+1][1:] :
                return False
        return True

sol = Solution()
sol.isToeplitzMatrix(A)
ans = all(a[:-1] == a1[1:] for a,a1 in zip(A, A[1:]))
ans = all(A[i][j] == A[i+1][j+1] for i in range(len(A)-1)
          for j in range(len(A[0])-1))
print(ans)
```

# 25 LC 867:  Transpose Matrix

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [transpose-matrix](https://leetcode.com/problems/transpose-matrix/)
![](../images/lc_867.png)
```python
class Solution:
    def transpose(self, A):
        # edge case
        if len(A) < 2: return A
        nrows, ncols = len(A), len(A[0])
        T = [[0] * nrows for _ in range(ncols)]
        for r in range(ncols): # one col has multiple rows
            for c in range(nrows):
                T[r][c] = A[c][r]
        return T

A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
ans = list(zip(*A))
ans = [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
np.transpose(A)
np.array(A).T
```

# 27 LC 985:  Sum of Even Numbers After Queries

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [sum-of-even-numbers-after-queries](https://leetcode.com/problems/sum-of-even-numbers-after-queries/)
![](../images/lc_985.png)
```python
class Solution:
    def sumEvenAfterQueries(self, A, queries):
        s = sum(i for i in A if i % 2 == 0) # even sum
        out = []
        for v, i in queries:
            if A[i] % 2 == 0: s -= A[i] # if even, subtract it
            A[i] += v
            if A[i] % 2 == 0: s += A[i] # if evan after change, add it
            out.append(s)
        return out

# Worst: time limit exceeded
class Solution:
    def sumEvenAfterQueries(self, A, queries):
        out = []
        for v,i in queries:
            A[i] = A[i] + v
            o = sum((i for i in A if i%2==0))
            out.append(o)
        return out

A = [1,2,3,4]
queries = [[1,0],[-3,1],[-4,0],[2,3]]
sol = Solution()
sol.sumEvenAfterQueries(A,queries)
```

# XX LC 1: Two Sum

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [two-sum](https://leetcode.com/problems/two-sum/)
![](../images/lc_1.png)
```python
# NOTE: We use hash lookup or split list into two parts.
# NOTE: [3,3].index(3) gives 0 but we might want 1
class Solution:
    def twoSum(self, nums, target):
        dic = {}
        for i, num in enumerate(nums):
            req = target - num # required number
            if req in dic:
                return [dic[req],i]
            dic[num] = i # dictionary key is number from nums and values is i
        return None

class Solution:
    def twoSum(self, nums, target):
        for i, num in enumerate(nums):
            req = target - num
            left = nums[:i]
            right = nums[i+1:]
            look = left + right
            if req in look:
                return [i, i+right.index(req)+1]
        return None

class Solution:
    def twoSum(self, nums, target):
        dic = {}
        for i, num in enumerate(nums):
            if num in dic:
                return [dic[num], i]
            else:
                dic[target - num] = i

class Solution:
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            req = target - nums.pop(0)
            if req in nums:
                return [i, nums.index(req)+i+1]

# just find true false using set
def find_sum_of_two(nums, target):
    seen_set = set()
    for num in nums:
        if target - num in seen_set:
            return True

        seen_set.add(num)
    return False

nums = [2, 7, 11, 15]; target = 9
nums = [4,4,4,4,4,4]; target = 8
nums = [3,3,2,5]; target = 8
nums = []; target = 8

sol = Solution()
sol.twoSum(nums,target)
```

# XX LC 784: Letter Case Permutation

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [letter-case-permutation](https://leetcode.com/problems/letter-case-permutation/)
![](../images/lc_784.png)
```python
class Solution:
    def letterCasePermutation(self, S):
        ans = [""]
        for s in S:
            print(f'\noutside s= {s}')
            if s.isdigit():
                print('digit: ans = {ans}')
                ans = [a+s for a in ans]
            else:
                tmp1 = [a+s.lower() for a in ans]
                tmp2 = [a+s.upper() for a in ans]
                print(f'else: ans = {ans}')
                print(f'tmp1 = {tmp1} ')
                print(f'tmp2 = {tmp2} ')
                ans = tmp1 + tmp2
        return ans
        return res

S = "a1b2"
sol = Solution()
sol.letterCasePermutation(S)

outside s= a
else: ans = ['']
tmp1 = ['a']
tmp2 = ['A']

outside s= 1
digit: ans = {ans}

outside s= b
else: ans = ['a1', 'A1']
tmp1 = ['a1b', 'A1b']
tmp2 = ['a1B', 'A1B']

outside s= 2
digit: ans = {ans}
['a1b2', 'A1b2', 'a1B2', 'A1B2']
#====================================DFS
class Solution:
    def letterCasePermutation(self, S):
        res = []
        def DFS(S, i):
            if i == len(S):
                res.append(S)
                return
            # For each i, Get upper and lowercase i of S
            # Then, when i==Lenght, it will be appended
            for case in {S[i].upper(), S[i].lower()}:
                DFS(S[:i]+case+S[i+1:], i+1)
        DFS(S,0)
        return res
```

# XX LC 412:  Fizz Buzz

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [fizz-buzz](https://leetcode.com/problems/fizz-buzz/)
![](../images/lc_412.png)
```python
class Solution:
    def fizzBuzz(self, n):
        ans = []
        for i in range(1,n+1):
            if i%15==0: ans.append('FizzBuzz')
            elif i%5==0: ans.append('Buzz')
            elif i%3==0: ans.append('Fizz')
            else: ans.append(str(i))

        return ans

Aliter:
[ 'Fizz' * (not i % 3) + 'Buzz' * (not i % 5 ) or str(i)
  for i in range(1, n+1) ]
```

# XX LC 88:  Merge Sorted Array (CTCI 10.1)

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [merge-sorted-array](https://leetcode.com/problems/merge-sorted-array/)
![](../images/lc_88.png)
```python
a,b,m,n = [2,5,8,20,30,50],[1,4,7],3,3
a,b,m,n = [0],[1],0,1 # special case a[:k+1] = b[:j+1]
a,b,m,n = [4,5,6,0,0,0],[1,2,3],3,3 # special case a[:k+1] = b[:j+1]

i=m-1
j=n-1
k=m+n-1
while i>=0 and j>=0: # look from last to 0th index
    if a[i]>b[j]:
        a[k]=a[i]
        i-=1
    else:
        a[k]=b[j]
        j-=1
    k-=1
if j>=0: # when all elements of b are smaller than a
    a[0:k+1]=b[0:j+1] # make first k elements of a all the elements from b

a
```
