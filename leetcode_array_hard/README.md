# LC 41: First Missing Positive
- https://leetcode.com/problems/first-missing-positive/
![](../images/lc_41.png)
```python
# O(n)
class Solution(object):
    def firstMissingPositive(self, nums):
        for i in range(len(nums)):
            while 0 <= nums[i]-1 < len(nums) and nums[nums[i]-1] != nums[i]:
                tmp = nums[i]-1
                nums[i], nums[tmp] = nums[tmp], nums[i]
        for i in range(len(nums)):
            if nums[i] != i+1:
                return i+1
        return len(nums)+1
        
#====================== O(nlogn)
class Solution(object):
    def firstMissingPositive(self, nums):
        nums.sort()
        res = 1
        for num in nums:
            if num == res:
                res += 1
        return res
        
#====================== O(n)
class Solution(object):
    def firstMissingPositive(self, nums):
        if not nums: return 1
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0: nums[i] = len(nums)+1
        for i in range(n):
            if abs(nums[i]) <= n: nums[abs(nums[i])-1] = -abs(nums[abs(nums[i])-1])
        for i in range(n):
            if nums[i] > 0: return i+1
        return n+1
```

# LC 51:  N-Queens
- https://leetcode.com/problems/n-queens/
![](../images/lc_51.png)
```python
class Solution:
    def solveNQueens(self, n):
        # define a helper function, depth first search
        def DFS(queens_curr, xy_diff, xy_sum):

            # queens to result, only when rows pointer p = n
            p = len(queens_curr)
            if p == n:
                result.append(queens_curr)
                return None

            
            # xy_diff is diagonal, and xy_sum is offdiagonal
            # go throught depth first, q is pointer for columns.
            for q in range(n):
                if  q     not in queens_curr and\
                    p - q not in xy_diff and\
                    p + q not in xy_sum:

                    # recursive call
                    DFS(queens_curr + [q],
                        xy_diff + [p - q],
                        xy_sum + [p + q]
                        )

        # call the helper function
        result = [] # this will be filled by DFS
        DFS([], [], [])
        
        return [ ["O" * i + "Q" + "O" * (n - i - 1) for i in res]
                 for res in result
               ]


sol = Solution()
res = sol.solveNQueens(4)
print('Number of possible solutions: {}'.format(len(res)))
print('\n'.join(res[0]))

Number of possible solutions: 2
OQOO
OOOQ
QOOO
OOQO

#================== same things, just writing the inner function outside
class Solution:
    def __init__(self):
        self.result = []

    def _DFS(self,n,queens_curr, xy_diff, xy_sum):
        # append queens to result only when rows pointer p == n
        p = len(queens_curr)
        if p == n:
            self.result.append(queens_curr)
            return None
        
        # xy_diff is diagonal, and xy_sum is offdiagonal
        # go throught depth first, q is pointer for columns.
        for q in range(n):
            if  q     not in queens_curr and\
                p - q not in xy_diff and\
                p + q not in xy_sum:

                # recursive call to itself
                self._DFS(n,
                    queens_curr + [q],
                    xy_diff     + [p - q],
                    xy_sum      + [p + q],
                    )
                    
    def solveNQueens(self, n):
        # call the helper function
        self._DFS(n, [], [], [])
        
        return [ ["O" * i + "Q" + "O" * (n - i - 1) for i in res]
                 for res in self.result
               ]
```
