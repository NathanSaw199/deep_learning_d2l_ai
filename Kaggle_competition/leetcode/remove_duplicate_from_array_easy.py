# implementation of a solution to remove duplicates from a sorted list of integers nums in-place. It modifies the list nums such that each element appears only once, and returns the number of unique elements, k.

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
       #The input list nums is sorted, so the first element (nums[0]) is guaranteed to be unique.Since the first unique element is already in its correct position, we start placing new unique elements from the second position (nums[1]), which corresponds to k = 1.The goal is to modify the list in-place without using additional memory.The pointer k ensures that all unique elements are grouped together at the start of the list. k starts from 1 because the first unique element (nums[0]) does not need to be moved.As the loop iterates through the list, whenever a new unique element is found (nums[i] != nums[i-1]), it is placed at the k-th position.
        k = 1
        #A loop starts from index 1 and iterates through the list. The index i is the current position being checked.
        for i in range(1, len(nums)):
            #If the current element nums[i] is different from the previous element nums[i - 1], it means nums[i] is a new unique value.
            if nums[i] != nums[i - 1]:
                #The unique value nums[i] is placed at the k-th position.
                nums[k] = nums[i]  
            #The pointer k is incremented by 1 to prepare for the next unique value.
                k += 1  
             
        #After the loop, k represents the count of unique elements in the list.The first k elements of nums now contain the unique values.
        return k


nums = [1,6, 1,1, 2,2,3]
solution = Solution()
k = solution.removeDuplicates(nums)
print("Modified nums:", nums[:k])
print("Number of unique elements:", k)