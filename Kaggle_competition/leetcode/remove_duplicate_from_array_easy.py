class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # acts as a pointer or index to place the next unique number in the nums list. 
        k = 1
        #This loop iterates from the second element to the end of the list. The index i tracks the current position in the list.
        for i in range(1, len(nums)):
            # If the current number is different from the previous unique number
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]  # Overwrite at index k
           
                k += 1  # Move the pointer
             
        
        return k

# Example usage
nums = [1, 1,1, 2,2]
solution = Solution()
k = solution.removeDuplicates(nums)
print("Modified nums:", nums[:k])
