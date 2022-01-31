<a id="top"></a>

Table of Contents
=================
   * [LC 274 h_index](#lc-274-h_index)
   * [LC 48 Rotate Image](#lc-48-rotate-image)
   * [LC 221: Maximal Square](#lc-221-maximal-square)
   * [LC 33: Search in Rotated Sorted Array](#lc-33-search-in-rotated-sorted-array)
   * [LC 81:  Search in Rotated Sorted Array II](#lc-81--search-in-rotated-sorted-array-ii)


# LC 274 h_index

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


https://leetcode.com/problems/h-index/
![](../images/lc_274.png)

```python
#============ best 32ms
from typing import List
class Solution:
    def hIndex(self, citations):
        n = len(citations)
        papers = [0] * (n + 1)  # papers[i] is the number of papers with i citations.
        for c in citations:
            papers[min(n, c)] += 1  # All papers with citations larger than n is count as n.
        i = n
        s = papers[n]  # sum of papers with citations >= i
        while i > s:
            i -= 1
            s += papers[i]
        return i

#====================== fast and easy
class Solution:
    def hIndex(self, c: List[int]) -> int:
    	L = len(c)
    	for i,j in enumerate(sorted(c)):
    		if L - i <= j: return L - i
    	return 0

#========================== inplace sorting
# fast 44 ms
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        l = len(citations)
        for i in range(l):
            if citations[i]>=l-i:
                return l-i
        return 0

# fast 48ms
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        dic = {}
        for c in citations:
            if c > len(citations):
                dic[len(citations)] = dic.get(len(citations),0)+1
            else:
                dic[c] = dic.get(c,0)+1

        count = 0
        for i in range(len(citations),-1,-1):
            count += dic.get(i,0)
            if count>=i:
                return i
        return 0


# slow 60ms
sum(i < j for i, j in enumerate(sorted(citations, reverse=True)))

# slow and long 60ms
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort( reverse = True )
        for idx, citation in enumerate(citations):

            # find the first index where citation is smaller than or equal to array index
            if idx >= citation:
                return idx

        return len(citations)
```

# LC 48 Rotate Image

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


https://leetcode.com/problems/rotate-image/
![](../images/lc_48.png)
```python
# O(n**2)
class Solution:
    def rotate(self, A):
        n = len(A)
        # columns ==> rows
        for i in range(n):
            for j in range(i):
                A[i][j], A[j][i] = A[j][i], A[i][j]

        # swap left and right in each rows (first to last etc.)
        for row in A:
            for j in range(n//2):
                row[j], row[~j] = row[~j], row[j]

matrix = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
]

"""
NOTE: In python for loop i and upto i gives Aij,Aji as end of diagonals except single elements.
*                   * upto n//2
1 2 3 column to row 1 4 7                   7 4 1
4 5 6 ==> c2r ==>   2 5 8  l2r colswap ==>  8 5 2
7 8 9               3 6 9                   9 6 3

"""

sol = Solution()
sol.rotate(matrix)
pprint(matrix,width=20)

#========================================= i,j two for loops, four values swap
class Solution:
    def rotate(self, A):
        n = len(A)
        for i in range(n//2):
            for j in range(n-n//2):
                # rotate top-left reversed clockwise elements to clockwise
                a = f'A[{i}][{j}], A[~{j}][{i}], A[~{i}][~{j}], A[{j}][~{i}] ='
                b = f'{A[i][j], A[~j][i], A[~i][~j], A[j][~i]}'
                print(a + b)
                A[i][j], A[~j][i], A[~i][~j], A[j][~i] = \
                         A[~j][i], A[~i][~j], A[j][~i], A[i][j]
A[0][0], A[~0][0], A[~0][~0], A[0][~0] =(1, 7, 9, 3)
A[0][1], A[~1][0], A[~0][~1], A[1][~0] =(2, 4, 8, 6)

#========================================== List comp
class Solution:
    def rotate(self, A):
        A[:] = [[row[i] for row in A[::-1]] for i in range(len(A))]

#========================================= zip and *
class Solution:
   def rotate(self, A):
       A[:] = zip(*A[::-1])

#========================================= numpy ro90 with k=3
import numpy as np
np.rot90(matrix,k=3)
```

# LC 221: Maximal Square

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- https://leetcode.com/problems/maximal-square/
- https://www.youtube.com/watch?v=FO7VXDfS8Gk
![](../images/lc_221.png)

```python
# O(m*n) space, one pass
class Solution:
    def maximalSquare(self, A):
        # edge case
        if not A:
            return 0
        r, c = len(A), len(A[0])
        dp = [[int(A[i][j]) for j in range(c)] for i in range(r)]
        res = max(max(dp)) # 1

        # start loop from 1: we dont need to do anything with outer rows and columns.
        # Logic: add bottom right corner value 1 to minimum of (left, up_left, up)
        for i in range(1, r): # 1 2 3
            for j in range(1, c): # 1 2 3 4
                dp[i][j] = (1 + min(dp[i][j-1],   # left
                                    dp[i-1][j-1], # up left
                                    dp[i-1][j]    # up
                                    )
                            )*int(A[i][j]) # becomes 0 if Aij is 0

                res = max(res, dp[i][j]**2)
        return res

A = [["1","0","1","0","0"],
     ["1","0","1","1","1"],
     ["1","1","1","1","1"],
     ["1","0","0","1","0"]]

# A = ["1"]
# A = []
sol = Solution()
sol.maximalSquare(A)
#==================================== O(n**2) time and O(row) space
class Solution:
    def maximalSquare(self, A):
        for i, r in enumerate(A):
            r = A[i] = list(map(int, r))
            for j, c in enumerate(r):
                if i * j * c:
                    r[j] = min(r[j-1], # left
                               A[i-1][j-1] # up left
                               A[i-1][j], # up
                               ) + 1 # r[j] is current position at row dp

        return max(map(max, A + [[0]])) ** 2
```

# LC 33: Search in Rotated Sorted Array

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>


- [LC 33: Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Python O(log N ) Solution with explanation, beats 98% time and space complexity](https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/338285/Python-O(log-N-)-Solution-with-explanation-beats-98-time-and-space-complexity)
- [Python3 modified BS explained and faster than 64.77%](https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/557476/Python3-modified-BS-explained-and-faster-than-64.77)
![](../images/lc_33.png)

NOTE:
- Use `m = l + (r-l)//2` to avoid overflow.
- We can also use faster method `m = l + r >>1` to get same thing.

```python
nums,target = [4,5,6,7,0,1,2], 4
nums,target = [4,5,6,7,0,1,2],7
nums,target = [4,5,6,7,0,1,2],1
nums,target = [4,5,6,7,0,1,2],2
nums,target = [6,7,0,1,2,4,5],4
nums,target = [6,7,0,1,2,4,5],7
nums,target = [],6

#======================================
# special binary search: recursive
class Solution:
    def search(self, nums, target):
        if not nums:
            return -1
        return self.binary_search(nums, target, 0, len(nums)-1)

    def binary_search(self, nums, target, start, end):
        if end < start:
            return -1
        mid = start + (end-start)//2
        if nums[mid] == target:
            return mid
        if nums[start] <= target < nums[mid]: # left side is sorted and has target
            return self.binary_search(nums, target, start, mid-1)
        elif nums[mid] < target <= nums[end]: # right side is sorted and has target
            return self.binary_search(nums, target, mid+1, end)
        elif nums[mid] > nums[end]: # right side is pivoted
            return self.binary_search(nums, target, mid+1, end)
        else: # left side is pivoted
            return self.binary_search(nums, target, start, mid-1)

# special binary search: iterative
class Solution:
    def search(self, nums, target) :
        if not nums: return -1
        # we change l so that in the end nums[l] will be target.
        l, r = 0, len(nums)-1
        while l<r:
            m = l + (r-l)//2
            if target == nums[m]: return m

            if nums[l] < nums[m]: # sorted left eg. [0,1,2,3,4,5,6,7]
                if nums[l] <= target <= nums[m]: r = m # target is at midleft
                else: l = m+1 # target is at midright

            # sorted right (pivoted)
            else: # eg. 6701 235 here nums[l] >= nums[m]
                if nums[m+1] <= target <= nums[r]: l = m+1 # target is at midright
                else: r = m
        return l if nums[l] == target else -1

# no binary search, no use of given info, just find index
# this is not solution, just the starting point.
class Solution:
    def search(self, nums, target) :
        if target in nums:
            return nums.index(target)
        else:
            return -1
```

# LC 81:  Search in Rotated Sorted Array II

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to TOC">Go to Top</a>

- [# LC 81:  Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
![](../images/lc_81.png)
```python
class Solution:
    def search(self, nums, target):
        l, r = 0, len(nums)-1
        while l <= r:
            mid = l + (r-l)//2

            # edge case
            if nums[mid] == target:
                return True

            # deal with dupes (we deal only left half using another while loop)
            while l < mid and nums[l] == nums[mid]:
                l += 1 # if we have many dupes, it will be O(N)

            # now it's modified binary search O(logN)
            # sorted left eg. [0,1,2,3,4,5,6,7]
            if nums[l] <= nums[mid]:
                # target is in the first half (decrease r from mid)
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1 # increase left from mid

            # sorted right (pivoted) 6701 2345 here nums[l] >= nums[m]
            else:
                # target is in the second half
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return False
```

