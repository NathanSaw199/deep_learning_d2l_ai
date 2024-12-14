class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        expected_num = []
        for i in nums:
            if i not in expected_num:
                expected_num.append(i)
        return expected_num
    

nums = [1,1,2]
solution = Solution()
k = solution.removeDuplicates(nums)
print(k)