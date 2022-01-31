Table of Contents
=================
   * [1 LC 977: Squares of a Sorted Array](#1-lc-977-squares-of-a-sorted-array)
   * [2 LC 344: Reverse String](#2-lc-344-reverse-string)
   * [3 LC 349: Intersection of Two Arrays](#3-lc-349-intersection-of-two-arrays)
   * [4 LC 283: Move Zeroes](#4-lc-283-move-zeroes)
   * [5 LC 167: Two Sum II - Input array is sorted](#5-lc-167-two-sum-ii---input-array-is-sorted)
   * [6 LC 350: Intersection of Two Arrays II](#6-lc-350-intersection-of-two-arrays-ii)
   * [7 LC 844: Backspace String Compare](#7-lc-844-backspace-string-compare)
   * [8 LC 26: Remove Duplicates from Sorted Array](#8-lc-26-remove-duplicates-from-sorted-array)
   * [9 LC 345: Reverse Vowels of a String](#9-lc-345-reverse-vowels-of-a-string)
   * [10 LC 88: Merge Sorted Array](#10-lc-88-merge-sorted-array)
   * [11 LC 125: Valid Palindrome](#11-lc-125-valid-palindrome)
   * [12 LC 28: Implement strStr()](#12-lc-28-implement-strstr)
   * [13 LC 538: K-diff Pairs in an Array](#13-lc-538-k-diff-pairs-in-an-array)

Note:
- while: 
  + Two sum sorted
  + Merge two sorted arrays ( two pointers m>0 and n>0)
- reversed loop: sorted squares

# 1 LC 977: Squares of a Sorted Array
- https://leetcode.com/problems/squares-of-a-sorted-array/
- https://leetcode.com/problems/squares-of-a-sorted-array/discuss/310865/Python%3A-A-comparison-of-lots-of-approaches!-Sorting-two-pointers-deque-iterator-generator
- https://medium.com/@george.seif94/this-is-the-fastest-sorting-algorithm-ever-b5cee86b559c
![](../images/lc_977.png)

```python
#==========================================
# Two end pointers reversed for loop
class Solution:
    def sortedSquares(self, A) :
        L = len(A)
        dp = [0] * L # for 1d array I can use[0]*L but for 2d we can not.
        l, r = 0, len(A) - 1

        # reversed iteration L=5 we need 4,3,2,1,0 L-1 is 4
	# and range upto -1 gives 0
        for i in range(L-1, -1, -1):
            if abs(A[l]) > abs(A[r]):
                dp[i] = A[l] ** 2
                l += 1
            else:
                dp[i] = A[r] ** 2
                r -= 1
        return dp
	
#============================== DP + Two end pointers while from end
class Solution:
    def sortedSquares(self, A) :
        # edge case
        if not A: return -1

        # length
        L = len(A)

        # init result
        dp = [0 for _ in range(L) ]

        # two pointers left and right
        l,r = 0, L-1

        # left and right square
        lsq,rsq = A[l] ** 2, A[r] ** 2
        
        # pointer to write from last
        i = L - 1
        while i >= 0:
            if lsq > rsq:
                dp[i] = lsq
                l += 1
                lsq = A[l] ** 2
            else:
                dp[i] = rsq
                r -= 1
                rsq = A[r] ** 2
            i -= 1
        return dp

#========================================== Inplace
class Solution:
    def sortedSquares(self, A) :
        res = [v**2 for v in A]
        res.sort() # inplace operation
        return res
        
#==========================================
# Built-in sorted (Timsort O(n) not nlogn) Fastest
class Solution:
    def sortedSquares(self, A) :
            return sorted([v**2 for v in A])
            
#==========================================
# Using collections.deque 
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

# 2 LC 344: Reverse String
https://leetcode.com/problems/reverse-string/
![](../images/lc_344.png)
```python
class Solution:
    def reverseString(self, s):
        for i in range(len(s)//2): # Only upto half
            s[i],s[~i] = s[~i],s[i] # in-place operation
            
 # simple ways
s[::-1] # s = list('hello')
s.reverse()
''.join(reversed(s))
```

# 3 LC 349: Intersection of Two Arrays
https://leetcode.com/problems/intersection-of-two-arrays/
![](../images/lc_349.png)
```python
class Solution:
    def intersection(self, A, B):
        dic = dict()
        ans = []
	
	# create dictionary from first list # {1: 0, 2: 0} from A = [1,2,2,1]; B = [2,2]
        for a in A:
            if a not in dic.keys():
                dic[a] = 0
		
	# check if elements of 2nd list is in the dictionary
        for b in B:
            if b in dic.keys() and b not in ans:
                ans.append(b)
        return ans
	
# Using built-in functions
nums1 = [1,2,2,1]; nums2 = [2,2]
list(set([i for i in nums1 if i in nums2]))
set(nums1).intersect(set(nums2)) # fastest
set(nums1) & set(nums2) # slightly slower
np.intersect1d(nums1,nums2)   # gives unique elements, [2] not [2,2]
```

# 4 LC 283: Move Zeroes
https://leetcode.com/problems/move-zeroes/
![](../images/lc_283.png)
```python
class Solution:
    def moveZeroes(self, nums):
            L = len(nums)
            left = 0
            
            # swap to left if element is non-zero
            for i in range(L):
                if nums[i] != 0: # nums[0] = 3 !=0, non_zero will increase
                    nums[left], nums[i] = nums[i], nums[left]
                    print(f'{i}: {nums} ** nums[{left}] goes to nums[{i}]')
                    left +=1
		    
Brute force solution: [i for i in nums if i!=0] + [i for i in nums if i==0]

#       *     * * *  not equal to zero
#       0 1 2 3 4 5
nums = [5,0,0,3,4,5]
sol = Solution()
sol.moveZeroes(nums)

0: [5, 0, 0, 3, 4, 5] ** nums[0] goes to nums[0]
3: [5, 3, 0, 0, 4, 5] ** nums[1] goes to nums[3]
4: [5, 3, 4, 0, 0, 5] ** nums[2] goes to nums[4]
5: [5, 3, 4, 5, 0, 0] ** nums[3] goes to nums[5]

#================================== Python enumerate
# 48ms
class Solution:
    def moveZeroes(self, nums) :
    	c = 0
    	for i, num in enumerate(nums):
    		if num != 0:
                 print(f'i={i} c={c} nums[i] = {nums[i]} i==c*num = {(i==c)*num}')
                 nums[c] =  num
                 nums[i] = (i==c)*num
                 c += 1
                 print(nums)
                 print()
    	return nums

nums = [3,12]
sol = Solution()
sol.moveZeroes(nums)
i=0 c=0 nums[i] = 3 i==c*num = 3
[3, 12]

i=1 c=1 nums[i] = 12 i==c*num = 12
[3, 12]
#================================== Two zero pointers and double while
class Solution:
    def moveZeroes(self, nums):
        L = len(nums)

        # two zero pointers
        l,r = 0,0

        while r < L:
            if nums[l] == 0:
                while r < L and nums[r] == 0: # e.g. when 0,0,1,0,5
                    r += 1
                if r > L-1:
                    break

                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r += 1
            else:
                l += 1
                r += 1
```

# 5 LC 167: Two Sum II - Input array is sorted
https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
![](../images/lc_167.png)
```python
#==================================================== Two pointers left and right
class Solution:
    def twoSum(self, nums, target):
        # edge case
        if not nums or target is None: return
        
        # two pointers
        l, r = 0, len(nums)-1 
        while l < r:
            mysum = nums[l] + nums[r]
            if mysum == target:  
                return [l+1, r+1]
            elif mysum > target:  # sum is big, make smaller
                r = r -1 
            else:  # sum is larger, make smaller
                l = l + 1 
        return
	
nums = [0,0,3,4]; target = 0; ans = (1,2) not (0,1)

NOTE: nums.index(i) gives the index of first i in nums.
It does not work when we have nums = [0,0,3,4]; targe=0;

#======================================================== Dictionary method
class Solution:
    def twoSum(self, numbers, target):
        # edge case
        if not numbers or target is None: return

        dic = dict()
        for i, num in enumerate(numbers): 
            if target - num in dic:
                return [dic[target-num]+1, i+1]  
            dic[num] = i 
        return 
```

# 6 LC 350: Intersection of Two Arrays II
https://leetcode.com/problems/intersection-of-two-arrays-ii/
![](../images/lc_350.png)
```python
#======================================================== Dictionary
class Solution:
    def intersect(self, nums1, nums2):
        dic = {} # counter dictionary
        ans = [] # NOTE: edge case empty nums1 nums1 is already covered

        # create counter dictionary from first list
        for n1 in nums1:
            if n1 not in dic:
                dic[n1] = 1
            else:
                dic[n1] += 1
        
	# use that dictioanry and decrease counts
        for n2 in nums2:
            if n2 in dic and dic[n2]>0:
                ans.append(n2)
                dic[n2] -= 1
        return ans
        
nums1 = [1,2,2,1]
nums2 = [2,2]

sol = Solution()
sol.intersect(nums1, nums2) # [2,2]
#======================================================== Two Pointers
class Solution:
    def intersect(self, nums1, nums2) :
            # edge case
            if not nums1 or not nums2: return []

            # sort O(nlogn)
            # if two lists are already sorted, then O(n+m)
            nums1.sort()
            nums2.sort()

            # two pointers for two lists
            p1, p2 = 0,0
            
            # answer
            ans = []
            
            # while loop for two pointers of two lists
            # nums1 = [1,2,2,1] ==> [1,1,2,2]
            # nums2 = [2,2]
            while p1 < len(nums1) and p2 < len(nums2):
                # second list elem smaller than first
                if nums2[p2] < nums1[p1]:
                    p2 += 1
                    continue

                elif nums2[p2] == nums1[p1]:
                    ans.append(nums2[p2])
                    p1 += 1
                    p2 += 1
                else: # 2nd list elem larger than first
                    p1 += 1
            return ans
            
class Solution:
    def intersect(self, nums1, nums2):
        res = []
         
        nums1.sort()
        nums2.sort()
        
        # two pointers
        p1, p2 = 0, 0 
        while p1 < len(nums1) and p2 < len(nums2):
            diff = nums1[p1] - nums2[p2]
            
            if diff == 0:
                res.append(nums1[p1])
                p1 += 1
                p2 += 1
    
            elif diff < 0:
                p1 += 1
    
            else:
                p2 += 1
        
        return res 

#======================================================== Counter
import collections
from collections import Counter

def intersect(self, nums1, nums2):      # nums1 = [1,2,2,1]; nums2 = [2,2,2]
    a, b = map(Counter, (nums1, nums2)) # a = Counter({1: 2, 2: 2}), b= Counter({2: 3})
    return list((a & b).elements())     # a&b = Counter({2: 2}) this is simply two elemetens of 2
    
def intersect(self, nums1, nums2):
    C = collections.Counter
    return list((C(nums1) & C(nums2)).elements())

def intersect(self, nums1, nums2) :
    return [*(Counter(nums1) & Counter(nums2)).elements()]
    
def intersect(self, nums1, nums2):           
    c1 = collections.Counter(nums1)
    c2 = collections.Counter(nums2)
    output = []
    for key in c1.keys() & c2.keys():
        output.extend([key]*min(c1[key], c2[key]))
    
    return output

#======================================================== List append remove (Extreme slow)
class Solution:
    def intersect(self, nums1, nums2):
        ans = []
        for n1 in nums1:
            if n1 in nums2:
                ans.append(n1)
                nums2.remove(n1)
        return ans
```

# 7 LC 844: Backspace String Compare
https://leetcode.com/problems/backspace-string-compare/
![](../images/lc_844.png)
```python
class Solution:
    def _backspace(self, string):
        ans = []
        for s in string:
            if s != '#': ans.append(s)
            elif s == '#' and ans: ans.pop() # aliter: del ans[-1]
        ans = ''.join(ans)
        return ans

    def backspaceCompare(self, S, T):
        return self._backspace(S) == self._backspace(T)

S = "ab#c" ; T = "ad#c"
sol = Solution()
sol.backspaceCompare(S,T)
```

# 8 LC 26: Remove Duplicates from Sorted Array
https://leetcode.com/problems/remove-duplicates-from-sorted-array/
![](../images/lc_26.png)
```python
Rem: Not equal to previous, increase and assign!!
Note: here s increases only when it sees unique values, 
      so in the end there are only s+1 unique values.
      
      2. Here for loop is from 1, so when we insert 2nd value (index 1)
         we need to s+=1 then make nums[s] = nums[i]
	 otherwise we might have stated with s=1, and s+=1 after number update.

class Solution:
    def removeDuplicates(self, nums):
        # edge case
        if not nums: 
            return -1
            
        # starting pointer
        s = 0
        print(' '*13, [0,1,2,3,4,5])
        for i in range(1, len(nums)):
            print('outside', i,s,':', nums)
            if nums[i] != nums[i-1]: # when NOT equal to previous, do not be confused here!!
                s +=1 # 1th element becomes unique number after 0th element, and so on.
                nums[s] = nums[i] # increase start and give value of ith.
                print(f'inside  {i} {s} : {nums} ** nums[{s}] becomes nums[{i}]')
                print()
        return s+1

#        *     *    here, 1st, 4th, 6th value are not equal to previous. Also note the updates in between.
#      0 1 2 3 4 5 6
nums =[0,1,1,1,2,2,3] # [], None
sol = Solution()
sol.removeDuplicates(nums)
nums

              [0, 1, 2, 3, 4, 5]
outside 1 0 : [0, 1, 1, 1, 2, 2, 3]
inside  1 1 : [0, 1, 1, 1, 2, 2, 3] ** nums[1] becomes nums[1]

outside 2 1 : [0, 1, 1, 1, 2, 2, 3]
outside 3 1 : [0, 1, 1, 1, 2, 2, 3]
outside 4 1 : [0, 1, 1, 1, 2, 2, 3]
inside  4 2 : [0, 1, 2, 1, 2, 2, 3] ** nums[2] becomes nums[4]

outside 5 2 : [0, 1, 2, 1, 2, 2, 3]
outside 6 2 : [0, 1, 2, 1, 2, 2, 3]
inside  6 3 : [0, 1, 2, 3, 2, 2, 3] ** nums[3] becomes nums[6]
```

# 9 LC 345: Reverse Vowels of a String
https://leetcode.com/problems/reverse-vowels-of-a-string/
![](../images/lc_345.png)
```python
class Solution:
    def reverseVowels(self,s):
        lst = list(s)
        vowels = 'aeiouAEIOU' 

        # Create list of indices of Vowels
        idx = []
        for i in range(len(s)):
            if s[i] in vowels:
                idx.append(i)

        # Swap only the vowel indices
        for i in range(len(idx)//2):
            lst[idx[i]], lst[idx[~i]] = lst[idx[~i]], lst[idx[i]] # note: lst[idx[i]] NOT lst[i]

        return ''.join(lst)
s = "leetcode"
sol = Solution()
sol.reverseVowels(s) # 'leotcede'

#========================================== Using two pointers
class Solution:    
    def reverseVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        l = 0
        r = len(s) - 1
        lst = list(s)
        
        while l < r:
            if lst[l] in vowels and lst[r] in vowels:
                lst[l], lst[r] = lst[r], lst[l]
                l += 1
                r -= 1
            elif lst[l] in vowels:
                r -= 1
            elif lst[r] in vowels:
                l += 1
            else:
                l += 1
                r -= 1
                
        return ''.join(lst)
```

# 10 LC 88: Merge Sorted Array
https://leetcode.com/problems/merge-sorted-array/
![](../images/lc_88.png)
```python
nums1 = [1,2,5,0,0]; m = 3
nums2 = [2,3]; n = 2
class Solution:
    def merge(self, nums1, m, nums2, n):
        # two pointers decreasing while loop
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]: # part1: compare valid last elements
                nums1[m + n - 1] = nums1[m - 1] # make last element of nums1
                print(f'if  : {m} {n} {nums1}')
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1] # part2: compare actual last elements
                print(f'else: {m} {n} {nums1}')
                n -= 1
	# outside of while loop
        nums1[:n] = nums2[:n] # n has already been decreased, usually to 0

# Few notes for while loop:
# for decreasing while loop both m and n usually decrease inside the code.
# At least one of m or n must be 0 to stop the while loop.
# Look for two situations where they decrease.

sol = Solution()
sol.merge(nums1,m,nums2,n)

nums1
if  : 3 2 [1, 2, 5, 0, 5]  ** last element becomes 5
else: 2 2 [1, 2, 5, 3, 5]  ** m has already decreased m+n-1 becomes 3
else: 2 1 [1, 2, 2, 3, 5]  ** n is decreased and new m+n-1 becomes 2
```

# 11 LC 125: Valid Palindrome
https://leetcode.com/problems/valid-palindrome/
![](../images/lc_125.png)
```python
class Solution:
    def isPalindrome(self, s):
        l, r = 0, len(s) - 1  # two pointers
        while l < r:
            if s[l].isalnum() and s[r].isalnum():
                if s[l].lower() != s[r].lower():  # ignoring cases
                    return False
                else:
                    l += 1
                    r -= 1
            elif not s[l].isalnum():  # left is not a alphanumeric
                l += 1
            else:  # right is not a alphanumeric
                r -= 1
        return True

s = "A man, a plan, a canal: Panama"
s = "02/02/2020" # acutally I did this question in Feb 02, 2020
sol = Solution()
if sol: print('true')

#====================== multiple while loops
class Solution:
    def isPalindrome(self, s):
        l, r = 0, len(s)-1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l <r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l +=1
            r -= 1
        return True
        
#============================ Clean s, then find palindrome
class Solution:
    def isPalindrome(self, s):
        ss = "".join(c for c in s if c.isalnum()).lower()
        l = 0
        r = len(ss) - 1
        while l < r:
            if ss[l] != ss[r]:
                return False
            l, r = l+1, r-1
        return True
        
#============================= Using list filter
class Solution:
    def isPalindrome(self, s):
        s = ''.join(list(filter(str.isalnum, str(s.lower()))))
        return s == s[::-1]
        
#=========================== Using re
class Solution:
    def isPalindrome(self, s):
        import re
        a = re.sub(r"\W+", r"", s.lower())
        return a == a[::-1]
```

# 12 LC 28: Implement strStr()
https://leetcode.com/problems/implement-strstr/
![](../images/lc_28.png)
```python
class Solution:
    def strStr(self, haystack, needle):
        H = len(haystack) # haystack = "hello" H = 5
        N = len(needle) # needle = 'll' # N = 2

        for i in range(H - N + 1): # H - N + 1 = 5 - 2 + 1 = 4 ==> 0,1,2,3
            if haystack[i: i + N] == needle: # 0 to 2 ==> 0,1 two letters
                return i
        return -1
        
#           01234
haystack = "hello"
needle = "ll" # needle = "l"  needle = ""
sol = Solution()
sol.strStr(haystack,needle)

#========================================= Similar but longer
class Solution:
    def strStr(self, haystack, needle):
        # edge case
        if needle == "":
            return 0

        H = len(haystack)
        N = len(needle)

        for i in range(H - N + 1):
            for j in range(N):
                if haystack[i+j] != needle[j]:
                    break
                if j == N-1:
                    return i
        return -1

#========================================= KMP O(m+n)
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
    
            if not haystack and not needle:
                return 0
            
            if not needle:
                return 0
            
            prefix_arr = [0]
            j, i, m, n = 0, 1, len(haystack), len(needle)
            while len(prefix_arr) != n:
                if needle[i] != needle[j] and j == 0:
                    prefix_arr.append(0)
                    i += 1
                elif needle[i] != needle[j] and j != 0:
                    j = prefix_arr[j - 1]
                else:
                    prefix_arr.append(j + 1)
                    i += 1
                    j += 1
            
            p1, p2 = 0, 0
            while p1 < m:
                if haystack[p1] == needle[p2]:
                    p1 += 1
                    p2 += 1
                    
                if p2 == n:
                    return p1 - p2
                elif p1 < m and haystack[p1] != needle[p2]:
                    if p2 == 0:
                        p1 += 1
                    else:
                        p2 = prefix_arr[p2 - 1]
            return -1
```

# 13 LC 538: K-diff Pairs in an Array
https://leetcode.com/problems/k-diff-pairs-in-an-array/
![](../images/lc_532.png)
```python
#============================================= Using sort, dict and set
class Solution:
    def findPairs(self, nums, k):
    	# edge case
        if k < 0: return 0
	
	# sort
        nums.sort()
	
	# allocate a set and a dict
        st = set() # set having two items
        dic = {}

        for num in nums:
            if num in dic:
                st.add( (dic[num], num ))
            dic[num + k] = num

        return len(st)
        
nums = [3, 1, 4, 1, 5] ; k = 2
sol = Solution()
sol.findPairs(nums,k)

#================================================= Using collections.Counter
from collections import Counter
class Solution:
    def findPairs(self, nums, k):
	# edge case
        if k < 0: return 0
        
        ctr = Counter(nums)
        st = set() # set having two items
        
        for num in ctr.keys():
	    # edge case, when k=0 we need duplicates
            if k == 0:
                if ctr[num] > 1:
                    st.add((num, num))
					
	    # add small,large number in set
            else:
                req = num + k # required complement number
                if req in ctr:
                    st.add((num, req) if num <= req else (req, num))
                    
        return len(st)
```
