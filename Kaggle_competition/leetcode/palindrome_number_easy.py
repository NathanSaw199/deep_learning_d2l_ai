class Solution(object):
    #he method takes one argument, x, which is the integer to be checked.
    def isPalindrome(self, x):
       #The integer x is converted to a string s so that it can be treated as a sequence of characters for easy comparison.
        s= str(x)
        #The length of the string s is determined and stored in the variable length.
        #length = len(s) returns the number of characters in the string s. 
        length = len(s)
        #A for loop iterates over the first half of the string, from index 0 to (length // 2) - 1.
        #length // 2 ensures that the loop covers only half the string, which is sufficient to check for a palindrome.
        #If length = 5 (odd length), length // 2 = 2 ensures that only the first two characters are compared with the last two characters.
        #If length = 6 (even length), length // 2 = 3 ensures the first three characters are compared with the last three. x = 121, s = "121", length = 3 and length // 2 = 1 and the loop will run once (i=0) and the first character will be compared with the last character. input x = 1221, s = "1221", length = 4 and length // 2 = 2 and the loop will run twice (i=0,1) and the first two characters will be compared with the last two characters.
        for i in range(length//2):
            #compare character from the start and end 
            #It compares s[i] (the character at position i from the start) with s[length - i - 1] (the character at position i from the end).For a string of length length, the last character is at index length - 1.To access the corresponding character from the end for index i (from the start), we subtract i + 1 from length.Input: x = 123.s = "123", length = 3.Iterations: i = 0: Compare s[0] ('1') with s[2] ('3'). They are not equal.returns False.
            #Input: x = 1221. s="1221". length =4 and iteration: i=0: compare s[0] which is 1 with s[4-0-1] which is s[3] which is 1. They are equal. i=1: compare s[1] which is 2 with s[4-1-1] which is s[2] which is 2. They are equal. The loop completes without returning False, so the number is a palindrome and the method returns True.
            if s[i] != s[length-1-i]:
                return False
        #If the loop completes without returning False, the number is a palindrome and the method returns True.
            return True
        
x = 121
solution = Solution()
print(solution.isPalindrome(x))



