# LC 1290: Convert Binary Number in a Linked List to Integer
https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/
![](../images/lc1290.png)
```python
class Solution:
    def getDecimalValue(self, head):
        """Get decimal value of binary numbers in a linked list.

        Parameters
        -----------
        head: Node
            Singly Linked List

        Returns
        --------
        ans: int
        """
        ans = 0
        # Read the binary number from MSB to LSB
        while head:
            ans = 2*ans + head.val
            head = head.next
        return ans

#======================== Create LinkedList to test the answer
class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

head = [1,0,1] # head is a list, make it singly linked list
nodes = [Node(i) for i in head]

for i in range(len(nodes)-1):
    nodes[i].next = nodes[i+1]

sol = Solution()
sol.getDecimalValue(nodes[0])

#=============================== Using int (base=2)
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        nums = []

        while head:
            nums.append(str(head.val))
            head = head.next

        return int("".join(nums), base=2)

#================================== Using Bit Manipulation << and |
class Solution(object):
    def getDecimalValue(self, head):
        ans = 0
        while head:
            ans <<= 1
            ans |= head.val
            head = head.next
        return ans
```