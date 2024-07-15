def find_max_weight_activities(activities):
    # Sort activities by start time
    activities.sort(key=lambda x: x[0])

    n = len(activities)
    dp = [0] * (n + 1)  # dp[i] will store the maximum weight of non-conflicting activities up to i
    activities.insert(0, (0, 0, 0))  # Add a dummy activity for easier indexing

    # Function to find the latest non-conflicting activity
    def latest_non_conflict(j):
        for i in range(j - 1, 0, -1):
            if activities[i][1] <= activities[j][0]:
                return i
        return 0

    # Base case
    dp[0] = 0  # With zero activities, the maximum weight is 0

    # Fill the DP table using the recursive formula
    for j in range(1, n + 1):
        incl_weight = activities[j][2] + dp[latest_non_conflict(j)]
        excl_weight = dp[j-1]
        dp[j] = max(incl_weight, excl_weight)

    # The cell that holds the solution
    return dp[n]

# Example usage:
# activities is a list of tuples (start_time, finish_time, weight)
activities = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)]
max_weight = find_max_weight_activities(activities)
print(f"The maximum weight of a set of non-conflicting activities is {max_weight}")



def max_contiguous_subsequence_sum(arr):
    if not arr:
        return 0

    n = len(arr)
    dp = [0] * n

    # Base case
    dp[0] = arr[0]
    
    for i in range(1, n):
        dp[i] = max(arr[i], arr[i] + dp[i-1])

    # The cell that holds the solution
    return max(dp)

# Example usage:
arr = [5, 15, -30, 10, -5, 40, 10]
max_sum = max_contiguous_subsequence_sum(arr)
print(f"The maximum sum of a contiguous subsequence is {max_sum}")




def min_penalty(hotels):
    # Number of hotels
    n = len(hotels)
    
    # Initialize DP array
    dp = [float('inf')] * (n + 1)
    
    # Starting point penalty is 0
    dp[0] = 0
    hotels.insert(0, 0)  # Insert the starting point at position 0
    
    # Fill the DP table using the recursive formula
    for i in range(1, n + 1):
        for j in range(i):
            dp[i] = min(dp[i], dp[j] + (300 - (hotels[i] - hotels[j]))**2)
    
    # The cell that holds the solution
    return dp[n]

# Example usage:
hotels = [100, 200, 400, 600, 700, 900]  # Example hotel mileposts
min_total_penalty = min_penalty(hotels)
print(f"The minimum total penalty for the trip is {min_total_penalty}")


