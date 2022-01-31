Table of Contents
=================
   * [Useful Notes](#useful-notes)
   * [Cytoolz](#cytoolz)
      * [accumulate](#accumulate)
      * [groupby](#groupby)
      * [interleave](#interleave)
      * [unique](#unique)
      * [frequencies (similar to Counter)](#frequencies-similar-to-counter)
      * [sliding_window](#sliding_window)
      * [partition_all](#partition_all)
      * [topk](#topk)
      * [pipe](#pipe)
      * [merge sorted](#merge-sorted)
      * [concat](#concat)
      * [merge and merge_with](#merge-and-merge_with)
      * [valmap](#valmap)
      * [keymap](#keymap)
      * [valfilter](#valfilter)
      * [keyfilter](#keyfilter)
   * [Useful Resources](#useful-resources)
   * [Few questions](#few-questions)

# Useful Notes
```python
#=========================================== Convensions and variable names
** Step1: First do brute force solution and ask would you like to improve the methods
** Step2: Ask what are the numbers, like can they be unsorted, can they be NULL, should I write unit tests?
** Step3: do only the inner part, then optimize (after brute force), then make function, then make class and add edge and comments. Say I like numpydoc docstring style.
** I can say, in python 3.8 we can use Type Hinting in functions. (`from Typing import List   a:List[int] -> float`
** Always test emtpy list, list with only one elemet, negative elements, unsorted elements and other edge cases.
** Step Last: Check for typos eg a1 instead of a1s or a1_sorted.
**
** Best Practices: Coding style  CamelCase for class, and small_letters for functions. (PEP8)


total_number = n BAD
total_number = N or L (GOOD)


s = starting pointer
i = running index
L = len(nums)

s = 0
for i in range(L): do something.
for num in nums: do something
for nn in nums: do something. (use double values for actual number, or simply use singluar element)

sm_diff ==> smallest_difference
curr_diff ==> current difference
l,r = left right
s,e = start end


#========================================= Error Check
assert isinstance(n,int), "n MUST be an integer"


#=============================================  Edge case
[] and [] == [] and [1,2] == [1,2] and [] ==> []

if [] and []:
    print('at least one is empty') # it will not print, but its correct
    
#============================================== Basic Python
indices = [[0,10],
           [1,11],
           [2,12]
          ]
for i,j in indices:
    print(i) # 0 1 2

#==================================== Modulo operator and integer division
4%2 = 0 
5%2 = 1  
if i%2 gives ODD (if True only condition works)

4//2 = 2
5//2 = 2 (floor of 2.5 is two)

#======================================== Initializing values
res = [0 for i in range(n)] ==>  [0]*n
if (n == 0) or (n==1)       ==> if n in [0,1]
dp[0],dp[1] = 0,0           ==> dp[:2] = [1,1]  (usually dp = [0] * (n+1), since we need last n)

dp[n] = dp[len(nums)] = dp[-1]
L = 3 # len(nums) or len(s) or length of something
dp = [0] * (L+1) ==>  [0,0,0,0] # four zeros

#========================================= Swapping two values
tmp = b
b = a+b
a = tmp

is same as a,b = b,a+b

#========================================== Itertools and cytoolz accumulate
from itertools import accumulate
nums = [1,2,3] # example for cumsum
list(accumulate(nums,lambda x,y: x+y)) # 1,3,6  # array first, func second

from cytoolz import accumulate
nums = [1,2,3]
list(accumulate(lambda x,y: x+y,nums)) # 1,3, 6 (func first, array second)

#========================================== collections deque
from collections import deque

dq = deque([1,2,3])
dq.append(4) # inplace operation
dq.appendleft(0)
last = dq.pop() # removes last element and returns it
first = dq.popleft() # removes first element and returns it

#=========================================== collections Counter
from collections import Counter

ctr = Counter([1,1,1,2,2,3,3,3,3,3]) # Counter({1: 3, 2: 2, 3: 5})
ctr.most_common(2)  # [(3, 5), (1, 3)]

```

# Cytoolz
- toolz is python implementation: https://toolz.readthedocs.io/en/latest/api.html
- cytoolz is C implementation: https://github.com/pytoolz/cytoolz

```python
import operator as op
import cytoolz as cz
```

## accumulate
```python
list(cz.accumulate(op.add, [1, 2, 3, 4, 5])) # [1, 3, 6, 10, 15] (this is cumsum)
```

## groupby
```python
iseven = lambda x: x % 2 == 0
cz.groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP

# {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}
```

## interleave
```python
list(cz.interleave([[1, 2], [3, 4]])) # [1, 3, 2, 4]
```

## unique
```python
tuple(cz.unique(['cat', 'mouse', 'dog', 'hen'], key=len)) # ('cat', 'mouse')
```

## frequencies (similar to Counter)
```python
cz.frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])  # {'cat': 3, 'ox': 1, 'pig': 2}
```

## sliding_window
```python
list(cz.sliding_window(2, [1, 2, 3, 4])) # [(1, 2), (2, 3), (3, 4)]
```

## partition_all
```python
list(cz.partition(2, [1, 2, 3, 4, 5], pad=None)) # [(1, 2), (3, 4), (5, None)]
list(cz.partition_all(2, [1, 2, 3, 4, 5])) # [(1, 2), (3, 4), (5,)]
```

## topk
```python
cz.topk(2, [1, 100, 10, 1000]) # (1000, 100) O(n logk)
```

## pipe
```python
double = lambda i: 2 * i
cz.pipe(3, double, str) # '6'
```

## merge sorted
```python

list(cz.merge_sorted([1, 3, 5], [2, 4, 6])) # [1, 2, 3, 4, 5, 6]
```

## concat
```python
list(cz.concat([[], [1], [2, 3]])) # [1, 2, 3]
```

## merge and merge_with
```python
cz.merge({1: 'one'}, {2: 'two'}) # {1: 'one', 2: 'two'}
cz.merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) # {1: 11, 2: 22}
```

## valmap
```python
bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
cz.valmap(sum, bills)  # {'Alice': 65, 'Bob': 45}
```

## keymap
```python
bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
cz.keymap(str.lower, bills)  # {'alice': [20, 15, 30], 'bob': [10, 35]}
```

## valfilter
```python
iseven = lambda x: x % 2 == 0
d = {1: 2, 2: 3, 3: 4, 4: 5}
cz.valfilter(iseven, d) # even values {1: 2, 3: 4}

cz.valfilter(lambda x: x%2, d) # odd values {2: 3, 4: 5}
cz.valfilter(lambda x: not x%2, d) # even values {1: 2, 3: 4}
```
## keyfilter
```python
iseven = lambda x: x % 2 == 0
d = {1: 2, 2: 3, 3: 4, 4: 5}
cz.keyfilter(iseven, d) # # even keys {2: 3, 4: 5}

cz.keyfilter(lambda x: x%2, d) # odd keys {1: 2, 3: 4}
# cz.keyfilter(lambda x: not x%2, d) # even keys {2: 3, 4: 5}
```


# Useful Resources
- [Medium 100+ Coding Interview Questions for Programmars](https://codeburst.io/100-coding-interview-questions-for-programmers-b1cf74885fb7)

# Few questions
1. What was the hardest bug you've faced?
