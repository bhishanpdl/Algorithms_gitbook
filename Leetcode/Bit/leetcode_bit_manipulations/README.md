Table of Contents
=================
   * [Notes](#notes)
   * [LC 1290: Convert Binary Number in a Linked List to Integer](#lc-1290-convert-binary-number-in-a-linked-list-to-integer)
   * [LC 461: Hamming Distance](#lc-461-hamming-distance)
   * [LC 476: Number Complement](#lc-476-number-complement)
   * [LC 136: Single Number](#lc-136-single-number)
   * [LC 762: Prime Number of Set Bits in Binary Representation](#lc-762-prime-number-of-set-bits-in-binary-representation)
   * [LC 784: Letter Case Permutation](#lc-784-letter-case-permutation)
   * [LC 693: Binary Number with Alternating Bits](#lc-693-binary-number-with-alternating-bits)
   * [LC 169: Majority Element](#lc-169-majority-element)
   * [LC 389: Find the Difference](#lc-389-find-the-difference)
   * [LC 371: Sum of Two Integers](#lc-371-sum-of-two-integers)
   * [LC 268: Missing Number](#lc-268-missing-number)
   * [LC 191: Number of 1 Bits](#lc-191-number-of-1-bits)
   * [LC 401: Binary Watch](#lc-401-binary-watch)
   * [LC 405: Convert a Number to Hexadecimal](#lc-405-convert-a-number-to-hexadecimal)
   * [LC 231: Power of Two](#lc-231-power-of-two)
   * [LC 342: Power of Four](#lc-342-power-of-four)
   * [LC 190: Reverse Bits](#lc-190-reverse-bits)

# Notes
- https://wiki.python.org/moin/BitwiseOperators
- https://wiki.python.org/moin/BitManipulation

![](../images/truth_table.png)

```python
# Bit manipulation
n & 1     n is odd
x << y   multiply x by 2**y
x >> y   divide   x by 2**y
x &  y   AND
x |  y   OR
x ^  y   X-OR

int('101',base=2) # 1**4 + 0*2 + 1*1 = 4 + 0 + 1 = 5
ord('A') = 65  chr(65) = 'A'
ord('a') = 97  chr(97) = 'a'



1.========= Numpydoc style docstring
https://stackoverflow.com/questions/3898572/what-is-the-standard-python-docstring-format
"""
Parameters
-----------
first: int
    first parameter is `first`

sec: {'value','other'}, optional
    Second parameter, by default 'value'

Returns
--------
ans : int
    Integer value answer

"""

```

# LC 461: Hamming Distance
https://leetcode.com/problems/hamming-distance
![](../images/lc461.png)
```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
```

# LC 476: Number Complement
https://leetcode.com/problems/number-complement
- https://leetcode.com/problems/number-complement/discuss/614121/Python-Fastest-One-Liner-(timed)

Complement Method

In programming, the two's complement of a number (`~num`) is defined as `-(num+1)`. So, the two's complement of 5 would be -6. The complement is with respect to `2^N` — that is, `num + ~num = 2^N` where N is the number of bits in our representation. In binary, this can be calculated by flipping all the bits and adding one. So if we convert 5 to binary, flip all the bits, and add them together, we should get `2^N-1`.
```
  101 (5)
+ 010 (2)
  ---
  111 (7)
+   1 (1)
  ---
 1000 (8)
```
From this, we can easily see that in order to find the complement of 5, we simply need to use the formula `2^N - 1 - num`, where N = the number of bits in 5 (in this case, 3). In code:
```
2**(len(bin(num))-2) - 1 - num
```
Note that since Python uses the `'0b'` prefix to represent a binary number, we subtract that from the length of the binary string for 5, which Python presents as `'0b101'`.

**Faster Complement with Bitshift**

We can calculate `2^N` much faster by bitshifting a 1. Since `2^N` in binary is just 1 followed by N zeroes (same as in decimal, where `10^N` is just 1 followed by N zeroes), we can replace `2^N` with `1<<N`. So now we have:
```
(1<<(len(bin(num))-2)) - 1 - num
```
And finally, as noted before, `~num = -(num+1)`, so we can replace this term in our code for elegance.
```
(1<<(len(bin(num))-2)) + ~num
```

```python
2**(len(bin(num))-2) - 1 - num
(1<<(len(bin(num))-2)) - 1 - num
(1<<(len(bin(num))-2)) + ~num

Brute force method
-------------------------
binaryNum = bin(num)[2:] # first two letters are 0b
binaryNumFlipped = ''
for i in binaryNum:
    binaryNumFlipped += '1' if i == 0 else '0'
return int(binaryNumFlipped, 2)
```

# LC 136: Single Number
https://leetcode.com/problems/single-number
![](../images/lc136.png)
```python
#===================== Using dictionary naive way
def singleNumber(nums):
    dic = {}
    for num in nums:
        dic[num] = dic.get(num, 0) + 1 # create a counter
    for key, val in dic.items():
        if val == 1:
            return key

#===================== Using Counter
from collections import Counter
def singleNumber(nums):
    return Counter(nums).most_common()[-1][0]

#===================== Using dictionary space O(1)
def singleNumber(nums) :
    dic = {}
    for num in nums:
        if not num in dic: dic[num] = 1
        else: del dic[num]
    return list(dic.keys())[0] # dict keys does not support indexing

#===================== Using set space O(1)
def singleNumber(nums):
    st = set()
    for num in nums:
        if not num in st: st.add(num)
        else: st.remove(num)
    return list(st)[0] # set does not support indexing

#===================== Using list space O(1)
def singleNumber(nums) :
    lst = []
    for num in nums:
        if not num in lst: lst.append(num)
        else: lst.remove(num)
    return lst[0]

#===================== Using XOR
def singleNumber(nums):
    res = 0
    for num in nums:
        res ^= num
    return res

#===================== Using sum and sets
def singleNumber(nums):
    return 2*sum(set(nums)) - sum(nums)

#===================== Using XOR and reduce
def singleNumber(nums):
    return functools.reduce(lambda x, y: x ^ y, nums)

#===================== Using XOR and reduce
def singleNumber(nums):
    return functools.reduce(operator.xor, nums)

https://leetcode.com/problems/single-number/discuss/468420/Python3-Bitwise-and-Non-Bitwise-Solutions-(w-Explanation)

>> x = 2  // 0010 in bits
>> x ^= 3 // 0010 ^ 0011 = 0001       (1) in bits
>> x ^= 3 // 0001 ^ 0011 = 0010       (2) in bits
>> x ^= 3 // 0010 ^ 0011 = 0001       (1) in bits
>> x ^= 3 // 0001 ^ 0011 = 0010       (2) in bits

As you see in the example, 
when we XOR 2 with 3 we get 1,
but if we do the operation again
then we end up with our original 2,
essentially undoing what we just did.
The same is true if we do this 
operation any even number of times.
```

# LC 762: Prime Number of Set Bits in Binary Representation
https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation
![](../images/lc762.png)
```python

```

# LC 784: Letter Case Permutation
https://leetcode.com/problems/letter-case-permutation
![](../images/lc784.png)
```python
#================================ One-liner with sum(1 for ...)
class Solution:
    def countPrimeSetBits(self, L, R) :
        return sum(1 for i in range(L,R+1)
                   if self._isPrime(bin(i).count('1')))

    # helper function
    import math
    def _isPrime(self, n):
        if (n % 2 == 0 and n > 2) or (n < 2):
            return False
        return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

L,R = 244, 269
sol = Solution()
sol.countPrimeSetBits(L,R) # 16
list(range(3,10))           # [3, 4, 5, 6, 7, 8, 9]
[11%i for i in range(3,10)] # [2, 3, 1, 5, 4, 3, 2]

#==================================== List comp space O(n)
class Solution:
    def countPrimeSetBits(self, L, R) :
        bins = [ bin(i).count('1') for i in range(L,R+1) ]
        primes = [ b for b in bins if self._isPrime(b)]
        return len(primes)

#==================================== Naive for loop space O(1)
class Solution:
    def countPrimeSetBits(self, L, R) :
        count = 0
        for i in range(L,R+1):
            x = 0
            for j in bin(i):
                if j == '1':
                    x += 1
            if self._isPrime(x): count += 1
        return count

```

# LC 693: Binary Number with Alternating Bits
https://leetcode.com/problems/binary-number-with-alternating-bits
![](../images/lc693.png)
```python

```

# LC 169: Majority Element
https://leetcode.com/problems/majority-element
![](../images/lc169.png)
```python

```

# LC 389: Find the Difference
https://leetcode.com/problems/find-the-difference
![](../images/lc389.png)
```python

```

# LC 371: Sum of Two Integers
https://leetcode.com/problems/sum-of-two-integers
![](../images/lc371.png)
```python

```

# LC 268: Missing Number
https://leetcode.com/problems/missing-number
![](../images/lc268.png)
```python

```

# LC 191: Number of 1 Bits
https://leetcode.com/problems/number-of-1-bits
![](../images/lc191.png)
```python

```

# LC 401: Binary Watch
https://leetcode.com/problems/binary-watch
![](../images/lc401.png)
```python

```

# LC 405: Convert a Number to Hexadecimal
https://leetcode.com/problems/convert-a-number-to-hexadecimal
![](../images/lc405.png)
```python

```

# LC 231: Power of Two
https://leetcode.com/problems/power-of-two
![](../images/lc231.png)
```python

```

# LC 342: Power of Four
https://leetcode.com/problems/power-of-four
![](../images/lc342.png)
```python

```

# LC 190: Reverse Bits
https://leetcode.com/problems/reverse-bits
![](../images/lc190.png)
```python

```
