class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # Pointer for the place to overwrite the next unique element
        k = 1 
        
        for i in range(1, len(nums)):
            # If the current number is different from the previous unique number
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]  # Overwrite at index k
           
                k += 1  # Move the pointer
             
        
        return k

# Example usage
nums = [1, 1, 2]
solution = Solution()
k = solution.removeDuplicates(nums)

print("Number of unique elements (k):", k)
print("Modified nums:", nums[:k])
